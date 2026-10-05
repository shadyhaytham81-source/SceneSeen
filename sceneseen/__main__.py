"""SceneSeen command line.

  python -m sceneseen analyze VIDEO [--clips] [--out DIR]
  python -m sceneseen evaluate [--split dev|test|all]
  python -m sceneseen tune
  python -m sceneseen train
  python -m sceneseen commercial VIDEO [--all]
  python -m sceneseen commercial-report [--min-confidence 0.4]
  python -m sceneseen check-data
  python -m sceneseen serve [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .config import ROOT, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "root", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_analyze(args, cfg) -> int:
    from .export import export_clips, write_scene_json
    from .pipeline import STAGES, analyze, public_result

    last = {}

    def progress(stage, frac):
        if last.get("stage") != stage:
            print(f"  · {STAGES[stage]}…", file=sys.stderr)
            last["stage"] = stage

    result = analyze(args.video, cfg, progress)
    out_dir = Path(args.out or cfg.paths.exports_dir / Path(args.video).stem)
    pub = public_result(result)
    jpath = write_scene_json(pub, out_dir / "scenes.json")
    t = result["debug"]["timings"]
    print(f"\n{result['scene_count']} scenes detected ({result['shot_count']} shots) in {Path(args.video).name}")
    for s in pub["scenes"]:
        print(f"  Scene {s['scene_id']:02d}  {_ts(s['start_seconds'])} → {_ts(s['end_seconds'])}  "
              f"({s['duration_seconds']:.1f}s)")
    from .media import probe as _probe
    from .pipeline import load_stage_outputs, unique_shots_for

    info = _probe(args.video)
    uniq = unique_shots_for(cfg, info, load_stage_outputs(cfg, info), result["scenes"])
    (out_dir / "unique_shots.json").write_text(json.dumps(uniq, indent=1), encoding="utf-8")
    u = uniq["summary"]
    print(f"\nUnique shots: {u['unique_shots']} of {u['original_shots']} "
          f"({u['repeated_shots']} repeated, {u['reduction'] * 100:.0f}% reduction) -> {out_dir / 'unique_shots.json'}")
    print(f"Scene data: {jpath}")
    print("Timings (s): " + ", ".join(f"{k}={v}" for k, v in t.items()))
    if args.debug:
        (out_dir / "debug.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"Debug data: {out_dir / 'debug.json'}")
    if args.clips:
        summary = export_clips(args.video, result["scenes"], out_dir / "clips", result["fps"], cfg.export)
        print(f"Clips: {out_dir / 'clips'} ({summary['copied']} stream-copied, {summary['reencoded']} re-encoded, "
              f"{summary['seconds']}s)")
    return 0


def _ts(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def cmd_evaluate(args, cfg) -> int:
    from .benchmark import NoLabelsError, format_report, load_dataset, run_evaluation

    try:
        data = load_dataset(cfg, args.split)
    except NoLabelsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    report = run_evaluation(data, cfg.grouping)
    md = format_report(report, args.split)
    print(md)
    rdir = ROOT / "reports"
    rdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (rdir / f"eval_{args.split}_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (rdir / f"eval_{args.split}_{stamp}.md").write_text(md, encoding="utf-8")
    print(f"Saved reports/eval_{args.split}_{stamp}.(json|md)")
    if args.split == "test":
        print("NOTE: this is the held-out test set. Do not tune thresholds after looking at these numbers.")
    return 0


def cmd_tune(args, cfg) -> int:
    from .benchmark import NoLabelsError, load_dataset, tune, write_tuned_config

    try:
        data = load_dataset(cfg, "dev")
    except NoLabelsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    res = tune(data, cfg.grouping)
    best = res.pop("_best")
    print(json.dumps(res, indent=1))
    loo, default = res["leave_one_out_mean"], res["default_dev_score"]
    if loo is None:
        print("\nWARNING: fewer than 3 dev videos, so generalisation cannot be estimated. Label more videos.")
    elif loo < default:
        print(f"\nVERDICT: tuned thresholds do NOT generalise (held-out F1 {loo:.3f} < untuned defaults {default:.3f}).\n"
              "Keep the defaults or label more dev videos before trusting tuned values.")
    else:
        print(f"\nVERDICT: tuning generalises on dev (held-out F1 {loo:.3f} >= defaults {default:.3f}).")
    out = Path(args.out)
    path = write_tuned_config(best, out, f"dev F1@2s={res['best_dev_score']:.3f} on {len(data)} dev videos "
                                         f"from {cfg.paths.ground_truth_dir.name}/")
    print(f"\nWrote {path}. Evaluate ONCE on the test set with:\n"
          f"  python -m sceneseen --config {out} evaluate --split test")
    return 0


def cmd_commercial(args, cfg) -> int:
    """Phase 2A: commercial objects + scene context for a video (Phase 1 is run/cached first)."""
    from .commercial import analysis as commercial_analysis
    from .media import probe
    from .pipeline import analyze, load_stage_outputs, unique_shots_for

    result = analyze(args.video, cfg)
    info = probe(args.video)
    stage = load_stage_outputs(cfg, info)
    unique = unique_shots_for(cfg, info, stage, result["scenes"])
    rec = commercial_analysis.analyze(
        stage["cache"].dir, info, stage["shots"], result["scenes"], unique, cfg.commercial,
        clip=stage["features"]["clip"], clip_model=(cfg.features.model, cfg.features.pretrained),
        cache_root=cfg.paths.cache_dir)
    out_dir = Path(args.out or cfg.paths.exports_dir / Path(args.video).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "commercial.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    s = rec["summary"]
    if rec["status"] == "unavailable":
        print(f"Commercial analysis unavailable: {rec['reason']}\n(Scenes and Unique Shots are unaffected.)")
        return 1
    print(f"\n{s['candidates_shown']} commercial objects in {s['scenes']} scenes "
          f"({s['unique_shots']} unique shots analysed instead of {s['original_shots']} shots; "
          f"{s['frames_inferred_now']} inferred now, {s['frames_from_cache']} from cache; status: {rec['status']})")
    for sc in rec["scenes"]:
        v, e = sc["context"]["venue"], sc["context"]["environment"]
        ctx = " · ".join(x for x in [v and f"{v['label']} ({v['confidence']:.0%})", e and e["label"]] if x) or "context unknown"
        print(f"\nScene {sc['scene_id']:02d}  {_ts(sc['start_seconds'])} → {_ts(sc['end_seconds'])}   [{ctx}]")
        for c in sc["candidates"]:
            if c["displayed"] or args.all:
                flag = "" if c["displayed"] else f"   (hidden: {c['hidden_reason']})"
                print(f"   {c['label']:<22} {c['category_name']:<24} confidence {c['detection_confidence']:.0%}  "
                      f"relevance {c['commercial_relevance']:.2f}  seen ×{c['seen_count']}{flag}")
    print(f"\nTimings (s): {rec['timings']}  ·  model: {rec['model'].get('name')} on {rec['model'].get('device', 'cache')}")
    print(f"Commercial data: {out_dir / 'commercial.json'}")
    return 0


def cmd_commercial_report(args, cfg) -> int:
    """Precision of reviewed commercial detections (from ground_truth/commercial_reviews/)."""
    from .commercial import review

    rep = review.summarize(cfg.paths.ground_truth_dir, args.min_confidence, cfg.commercial.min_relevance,
                           cfg.commercial.min_confidence)
    d = rep["displayed"]
    if not rep["all_reviewed"]["reviewed"]:
        print("No reviews yet. In the web app: Developer mode → Commercial Objects → mark detections Correct / Wrong.")
        return 1
    scope = f"at a flat confidence ≥ {args.min_confidence}" if args.min_confidence is not None else "under the current display rules"
    print(f"Displayed detections {scope}: precision {d['precision']:.1%} "
          f"({d['correct']} correct, {d['wrong']} wrong, {rep['videos']} videos); missed objects reported: {rep['missed_reported']}")
    for title, key in (("By category", "by_category"), ("By object type", "by_type"), ("By video", "by_video")):
        print(f"\n{title}")
        for k, v in sorted(rep[key].items(), key=lambda kv: -kv[1]["reviewed"]):
            print(f"  {k[:44]:<44} {v['precision']:.0%}  ({v['correct']}/{v['reviewed']})")
    print("\nPrecision vs confidence threshold (all reviewed detections, incl. hidden ones):")
    for r in rep["threshold_sweep"]:
        if r["reviewed"]:
            print(f"  ≥ {r['min_confidence']:.2f}: {r['precision']:.1%}  ({r['correct']}/{r['reviewed']})")
    return 0


def cmd_train(args, cfg) -> int:
    """Train the optional boundary classifier on DEV, validate on VAL, never touch TEST."""
    from . import learning as L
    from .benchmark import NoLabelsError, load_dataset
    from .ground_truth import label_files, load_or_create_split, nfc

    try:
        dev = load_dataset(cfg, "dev")
    except NoLabelsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        val = load_dataset(cfg, "val")
    except NoLabelsError:
        val = []
    split = load_or_create_split(cfg.paths.ground_truth_dir, [nfc(p.stem) for p in label_files(cfg.paths.ground_truth_dir)])
    L.assert_no_test_leak([it["stem"] for it in dev], split)
    print(f"training videos (dev): {len(dev)} | validation videos (val): {len(val)} | "
          f"final-test videos: {len(split['test'])} (not loaded)")
    if not val:
        print("WARNING: no validation videos; a model cannot be accepted without held-out validation.")
    rep = L.run_experiment(dev, val, cfg.grouping, with_tuned=not args.no_tuned)
    decision = L.decide(rep) if val else {"any_accepted": False, "methods": {}, "baseline": {}}
    md = L.format_experiment(rep, decision) if val else "(no validation set)\n"
    examples = [L.build_examples(it, cfg.grouping) for it in dev]
    model, hp = L.train_model(examples)
    accepted = bool(decision["methods"].get("C_logreg", {}).get("accepted", False))
    model.meta = L.training_metadata(dev, examples, hp, cfg.grouping, {
        "validation_videos": [it["stem"] for it in val],
        "metrics": {"dev_leave_one_video_out": rep["summary"]["dev_loo"]["C_logreg"],
                    "validation": rep["summary"]["val"]["C_logreg"],
                    "current_rule_dev_leave_one_video_out": rep["summary"]["dev_loo"]["A_current_rule"],
                    "current_rule_validation": rep["summary"]["val"]["A_current_rule"]},
        "accepted": accepted,
        "decision_rule": "better than the current rule on dev leave-one-video-out AND not worse on validation "
                         "AND better on more videos than worse",
    })
    body = model.to_dict()
    out = Path(args.out) / f"boundary_logreg_{body['artifact_hash']}.json"
    model.save(out)
    rdir = ROOT / "reports"
    rdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (rdir / f"learning_{stamp}.md").write_text(md, encoding="utf-8")
    (rdir / f"learning_{stamp}.json").write_text(json.dumps({"report": rep, "decision": decision}, indent=1, default=str),
                                                 encoding="utf-8")
    print(md)
    print(f"Model artifact: {out}  (hash {body['artifact_hash']}, ground-truth version {body['ground_truth_version']})")
    if accepted:
        print(f"ACCEPTED. To use it, set in a config file:\n  [boundary_model]\n  path = \"{out.relative_to(ROOT)}\"")
    else:
        print("NOT ACCEPTED: the learned model does not beat the current rule on held-out data. "
              "SceneSeen keeps using the rule; the artifact is saved for the record only.")
    return 0


def cmd_check_data(args, cfg) -> int:
    """List every ground-truth label and whether its video is present, readable and identical
    to the labelled file (size always; SHA-256 with --verify)."""
    import hashlib
    import json as _json

    from .ground_truth import find_video, label_files, load_label, nfc
    from .media import VideoError, is_dataless, probe

    gt, vids = cfg.paths.ground_truth_dir, cfg.paths.videos_dir
    mpath = gt / "videos_manifest.json"
    manifest = {nfc(v["video"]): v for v in _json.loads(mpath.read_text(encoding="utf-8"))["videos"]} \
        if mpath.exists() else {}
    labels = label_files(gt)
    print(f"labels: {gt}\nvideos: {vids}\n")
    ok = 0
    for lp in labels:
        name = _json.loads(lp.read_text(encoding="utf-8")).get("video", "?")
        vp = find_video(vids, name)
        expect = manifest.get(nfc(name))
        status = "MISSING"
        if vp.exists() and is_dataless(vp):
            status = "cloud-only placeholder (download it first)"
        elif vp.exists():
            try:
                info = probe(vp)
                n = len(load_label(lp, info.duration)["boundaries"])
                status = f"ok ({info.duration / 60:.1f} min, {n} boundaries)"
                if expect and vp.stat().st_size != expect["bytes"]:
                    status = (f"DIFFERENT FILE: {vp.stat().st_size} bytes, expected {expect['bytes']} "
                              f"(another download/quality shifts timestamps)")
                elif expect and args.verify:
                    h = hashlib.sha256()
                    with open(vp, "rb") as f:
                        for chunk in iter(lambda: f.read(8 << 20), b""):
                            h.update(chunk)
                    status += ", sha256 match" if h.hexdigest() == expect["sha256"] else ""
                    if h.hexdigest() != expect["sha256"]:
                        status = "DIFFERENT FILE: sha256 mismatch"
            except (VideoError, ValueError) as e:
                status = f"ERROR: {e}"
        print(f"  [{'x' if status.startswith('ok') else ' '}] {name}\n        {status}")
        ok += status.startswith("ok")
    print(f"\n{ok}/{len(labels)} labelled videos ready.")
    if ok < len(labels):
        print("\nThe dataset videos are NOT stored in Git (large, copyrighted). Get them from the project owner,\n"
              f"copy them into {vids} with exactly the names above, then re-run:\n"
              "  python -m sceneseen check-data --verify\n"
              "See data/README.md. (To analyse NEW videos you don't need this: upload them in the web app.)")
    return 0 if ok == len(labels) else 1


def cmd_serve(args, cfg) -> int:
    import os

    import uvicorn

    if args.config:
        os.environ["SCENESEEN_CONFIG"] = os.pathsep.join(str(Path(c).resolve()) for c in args.config)
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sceneseen", description="SceneSeen Phase 1 — video scene segmentation")
    p.add_argument("--config", action="append", default=[],
                   help="TOML file(s) overriding config/default.toml (repeatable, later wins)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="detect scenes in a video")
    a.add_argument("video")
    a.add_argument("--out", help="output directory (default data/exports/<video stem>)")
    a.add_argument("--clips", action="store_true", help="also export one clip per scene")
    a.add_argument("--debug", action="store_true", help="also write shots/scores/decisions as debug.json")
    e = sub.add_parser("evaluate", help="score predictions against ground_truth/ labels")
    e.add_argument("--split", choices=["dev", "val", "test", "all"], default="dev")
    t = sub.add_parser("tune", help="grid-search grouping thresholds on the dev split")
    t.add_argument("--out", default=str(ROOT / "config" / "tuned.toml"), help="where to write the tuned [grouping]")
    co = sub.add_parser("commercial", help="Phase 2A: commercial objects + scene context for a video")
    co.add_argument("video")
    co.add_argument("--out", help="output directory (default data/exports/<video stem>)")
    co.add_argument("--all", action="store_true", help="also list candidates hidden from the normal view")
    cr = sub.add_parser("commercial-report", help="precision of reviewed commercial detections")
    cr.add_argument("--min-confidence", type=float, default=None,
                    help="re-evaluate existing reviews at this display threshold")
    tr = sub.add_parser("train", help="train the optional boundary classifier (dev -> validate on val; never test)")
    tr.add_argument("--out", default=str(ROOT / "models"), help="directory for the versioned model artifact")
    tr.add_argument("--no-tuned", action="store_true", help="skip the (slow) tuned-rule comparison")
    c = sub.add_parser("check-data", help="check that every ground-truth label has its video in data/videos/")
    c.add_argument("--verify", action="store_true", help="also compare SHA-256 with ground_truth/videos_manifest.json")
    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(*args.config)
    return {"analyze": cmd_analyze, "evaluate": cmd_evaluate, "tune": cmd_tune, "serve": cmd_serve,
            "check-data": cmd_check_data, "train": cmd_train,
            "commercial": cmd_commercial, "commercial-report": cmd_commercial_report}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
