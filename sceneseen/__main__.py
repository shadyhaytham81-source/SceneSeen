"""SceneSeen command line.

  python -m sceneseen analyze VIDEO [--clips] [--out DIR]
  python -m sceneseen evaluate [--split dev|test|all]
  python -m sceneseen tune
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
    print(f"\nScene data: {jpath}")
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


def cmd_check_data(args, cfg) -> int:
    """List every ground-truth label and whether its video is present and readable."""
    import json as _json

    from .ground_truth import find_video, load_label
    from .media import VideoError, is_dataless, probe

    gt, vids = cfg.paths.ground_truth_dir, cfg.paths.videos_dir
    labels = sorted(p for p in gt.glob("*.json") if p.name != "splits.json")
    print(f"labels: {gt}\nvideos: {vids}\n")
    ok = 0
    for lp in labels:
        name = _json.loads(lp.read_text(encoding="utf-8")).get("video", "?")
        vp = find_video(vids, name)
        status = "MISSING"
        if vp.exists() and is_dataless(vp):
            status = "cloud-only placeholder (download it first)"
        elif vp.exists():
            try:
                info = probe(vp)
                load_label(lp, info.duration)
                status, ok = f"ok ({info.duration / 60:.1f} min, {len(_json.loads(lp.read_text(encoding='utf-8'))['boundaries'])} boundaries)", ok + 1
            except (VideoError, ValueError) as e:
                status = f"ERROR: {e}"
        print(f"  [{'x' if status.startswith('ok') else ' '}] {name}\n        {status}")
    print(f"\n{ok}/{len(labels)} labelled videos ready.")
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
    e.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    t = sub.add_parser("tune", help="grid-search grouping thresholds on the dev split")
    t.add_argument("--out", default=str(ROOT / "config" / "tuned.toml"), help="where to write the tuned [grouping]")
    sub.add_parser("check-data", help="check that every ground-truth label has its video in data/videos/")
    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(*args.config)
    return {"analyze": cmd_analyze, "evaluate": cmd_evaluate, "tune": cmd_tune, "serve": cmd_serve,
            "check-data": cmd_check_data}[args.cmd](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
