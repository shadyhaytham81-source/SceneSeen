"""Evaluation runner and threshold tuning over labelled videos.

Protocol (no data leakage):
* ground_truth/splits.json fixes a dev/test split (created once, then frozen).
* `tune` searches grouping parameters on DEV videos only and writes config/tuned.toml.
  It also reports a leave-one-video-out estimate on dev, which shows how much the
  chosen thresholds overfit.
* `evaluate --split test` scores the frozen config on TEST videos, once, at the end.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import grouping
from .config import Config, GroupingConfig, config_hash
from .evaluation import aggregate, evaluate
from .ground_truth import (ROLES, _local_upload_copy, find_video, label_files, load_label, load_or_create_split,
                           nfc)
from .media import VideoInfo, is_dataless, probe
from .pipeline import load_stage_outputs

log = logging.getLogger(__name__)

TOLERANCES = (1.0, 2.0, 3.0)
OBJECTIVE_TOL = "2"  # tuning objective: macro boundary F1 at ±2 s

METHODS = ("baseline_all_shots", "baseline_adjacent", "sceneseen")


class NoLabelsError(RuntimeError):
    pass


def cached_info(cfg: Config, name: str) -> VideoInfo | None:
    """VideoInfo from a previous analysis of this file (matched by its recorded name), provided
    shots and features for the CURRENT config are in the cache. Lets evaluation and training run
    when the video itself is unavailable (e.g. evicted to iCloud); nothing needs decoding."""
    shots_key = config_hash(cfg.shots)
    feat_key = f"{shots_key}_{config_hash(cfg.features)}"
    target = nfc(name)
    for d in sorted(cfg.paths.cache_dir.glob("*/")):
        info_p = d / "info.json"
        if not info_p.exists():
            continue
        names = set()
        up = d / "upload.json"
        if up.exists():
            names.add(nfc(json.loads(up.read_text(encoding="utf-8")).get("name", "")))
        meta = json.loads(info_p.read_text(encoding="utf-8"))
        names.add(nfc(Path(meta.get("path", "")).name))
        if target in names and (d / f"shots_{shots_key}.json").exists() and (d / f"features_{feat_key}.npz").exists():
            return VideoInfo(**meta)
    return None


def resolve_video(cfg: Config, name: str) -> tuple[VideoInfo | None, str]:
    """(info, source) with source in video | upload-copy | cache | missing."""
    vp = find_video(cfg.paths.videos_dir, name)
    if vp.exists() and not is_dataless(vp):
        return probe(vp), "video"
    alt = _local_upload_copy(name, cfg.paths.cache_dir)
    if alt is not None:
        return probe(alt), "upload-copy"
    info = cached_info(cfg, name)
    return (info, "cache") if info else (None, "missing")


def load_dataset(cfg: Config, split: str) -> list[dict]:
    """Labelled videos for a role (dev | val | test | all), with shots/features loaded from cache
    or computed. Labels without a role in splits.json are never returned."""
    labels = label_files(cfg.paths.ground_truth_dir)
    if not labels:
        raise NoLabelsError(
            f"no labelled videos found: put videos in {cfg.paths.videos_dir} and labels in "
            f"{cfg.paths.ground_truth_dir} (see docs/EVALUATION.md)")
    sp = load_or_create_split(cfg.paths.ground_truth_dir, [nfc(p.stem) for p in labels])
    wanted = {s for r in ROLES for s in sp[r]} if split == "all" else set(sp.get(split, []))
    data = []
    for lp in labels:
        stem = nfc(lp.stem)
        if stem not in wanted:
            continue
        name = json.loads(lp.read_text(encoding="utf-8")).get("video", "")
        info, source = resolve_video(cfg, name)
        if info is None:
            log.warning("skipping %s: video not found in %s and no cached analysis", lp.name, cfg.paths.videos_dir)
            continue
        if source != "video":
            log.warning("%s: video file unavailable, using %s", name[:40], source)
        label = load_label(lp, info.duration)
        t = time.perf_counter()
        stage = load_stage_outputs(cfg, info)
        role = next(r for r in ROLES if stem in sp[r])
        data.append({"stem": stem, "info": info, "gt": label["boundaries"], "stage": stage, "role": role,
                     "source": source, "label_path": lp, "load_seconds": round(time.perf_counter() - t, 2)})
        log.info("loaded %s (%d shots, %d labelled boundaries)", stem, len(stage["shots"]), len(label["boundaries"]))
    if not data:
        raise NoLabelsError(f"split '{split}' has no usable videos (see {cfg.paths.ground_truth_dir}/splits.json; "
                            f"videos go in {cfg.paths.videos_dir})")
    return data


def predict(item: dict, method: str, g: GroupingConfig) -> list[float]:
    shots, f, dur = item["stage"]["shots"], item["stage"]["features"], item["info"].duration
    if method == "baseline_all_shots":
        return grouping.baseline_all_shots(shots)
    if method == "baseline_adjacent":
        return grouping.baseline_adjacent(shots, f["clip"], f["color"], dur, g)
    if method == "sceneseen":
        return grouping.group_shots(shots, f["clip"], f["color"], dur, g)["boundaries"]
    raise ValueError(method)


def run_evaluation(data: list[dict], g: GroupingConfig, methods=METHODS) -> dict:
    report: dict = {"grouping_config": asdict(g), "methods": {}}
    for m in methods:
        per = []
        for it in data:
            ev = evaluate(predict(it, m, g), it["gt"], it["info"].duration, TOLERANCES)
            ev["video"] = it["stem"]
            per.append(ev)
        report["methods"][m] = {"aggregate": aggregate(per), "per_video": per}
    report["performance"] = [
        {"video": it["stem"], "duration": it["info"].duration, "shots": len(it["stage"]["shots"]),
         **it["stage"]["timings"]} for it in data]
    return report


# ---------------------------------------------------------------- tuning

SEARCH_SPACE = {
    "window_shots": [2, 4, 6, 8],
    "coherence_topk": [1, 2, 3],
    "abs_threshold": [0.3, 0.4, 0.45, 0.5, 0.6],
    "depth_threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
    "strong_threshold": [0.65, 0.75, 1.01],
    "min_scene_seconds": [4.0, 6.0, 10.0],
    "clip_weight": [1.0, 0.75, 0.5],
}


def _grid(base: GroupingConfig):
    keys = list(SEARCH_SPACE)
    for values in itertools.product(*(SEARCH_SPACE[k] for k in keys)):
        kw = dict(zip(keys, values))
        kw["color_weight"] = 1.0 - kw["clip_weight"]
        yield base.replace(**kw)


_SCORE_MEMO: dict = {}


def _fast_predict(it: dict, g: GroupingConfig) -> list[float]:
    """Same result as predict(..., "sceneseen"), but caches the coherence scores, which only
    depend on similarity/window params, so threshold sweeps cost almost nothing."""
    key = (it["stem"], g.clip_weight, g.color_weight, g.clip_floor, g.window_shots, g.window_seconds,
           g.distance_penalty, g.coherence_topk)  # depth_mode only affects selection, not these scores
    if key not in _SCORE_MEMO:
        f, shots = it["stage"]["features"], it["stage"]["shots"]
        S = grouping.similarity_matrix(f["clip"], f["color"], g)
        _SCORE_MEMO[key] = grouping.cut_scores(S, shots, g) if len(shots) > 1 else []
    scores = [dataclasses.replace(c) for c in _SCORE_MEMO[key]]
    return [c.time for c in grouping.select_boundaries(scores, it["info"].duration, g)]


def _objective(data: list[dict], g: GroupingConfig) -> float:
    f1s = [evaluate(_fast_predict(it, g), it["gt"], it["info"].duration, (float(OBJECTIVE_TOL),))
           ["by_tolerance"][OBJECTIVE_TOL]["f1"] for it in data]
    return float(np.mean(f1s))


def _best(data: list[dict], grid: list[GroupingConfig]) -> tuple[GroupingConfig, float]:
    best, best_s = grid[0], -1.0
    for g in grid:
        s = _objective(data, g)
        if s > best_s:
            best, best_s = g, s
    return best, best_s


def tune(data: list[dict], base: GroupingConfig) -> dict:
    grid = list(_grid(base))
    t = time.perf_counter()
    best, score = _best(data, grid)
    loo = []
    if len(data) >= 3:  # leave-one-video-out estimate of generalisation
        for i in range(len(data)):
            rest = data[:i] + data[i + 1:]
            g, _ = _best(rest, grid)
            loo.append({"held_out": data[i]["stem"], "f1": _objective([data[i]], g)})
    return {
        "objective": f"macro boundary F1 @ ±{OBJECTIVE_TOL}s on dev",
        "grid_size": len(grid),
        "best_config": asdict(best),
        "best_dev_score": score,
        "default_dev_score": _objective(data, base),
        "leave_one_out": loo,
        "leave_one_out_mean": float(np.mean([x["f1"] for x in loo])) if loo else None,
        "seconds": round(time.perf_counter() - t, 1),
        "_best": best,
    }


def write_tuned_config(g: GroupingConfig, path: Path, note: str) -> Path:
    lines = [f"# Generated by `python -m sceneseen tune` — {note}", "[grouping]"]
    for k, v in asdict(g).items():
        lines.append(f"{k} = {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- reporting

def format_report(report: dict, split: str) -> str:
    rows = [f"## SceneSeen evaluation — split: {split}", "",
            "| method | F1@1s | F1@2s | F1@3s | P@2s | R@2s | timing err (s) | coverage | overflow | F_CO | pred/true scenes |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in report["methods"].items():
        a = r["aggregate"]
        bt = a["by_tolerance"]
        err = bt["2"]["mean_abs_error"]
        rows.append(
            f"| {m} | {bt['1']['macro_f1']:.3f} | {bt['2']['macro_f1']:.3f} | {bt['3']['macro_f1']:.3f} | "
            f"{bt['2']['micro_precision']:.3f} | {bt['2']['micro_recall']:.3f} | "
            f"{'-' if err is None else f'{err:.2f}'} | {a['coverage']:.3f} | {a['overflow']:.3f} | "
            f"{a['f_co']:.3f} | {a['count_ratio']:.2f} |")
    rows += ["", "F1 = macro average over videos; P/R pooled over videos. pred/true > 1 means over-segmentation.", "",
             "| video | duration (s) | shots | true scenes | predicted scenes | F1@2s |", "|---|---|---|---|---|---|"]
    per = report["methods"]["sceneseen"]["per_video"]
    perf = {p["video"]: p for p in report["performance"]}
    for v in per:
        rows.append(f"| {v['video']} | {perf[v['video']]['duration']:.0f} | {perf[v['video']]['shots']} | "
                    f"{v['n_gt_scenes']} | {v['n_pred_scenes']} | {v['by_tolerance']['2']['f1']:.3f} |")
    return "\n".join(rows) + "\n"
