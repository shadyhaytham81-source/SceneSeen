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
from .config import Config, GroupingConfig
from .evaluation import aggregate, evaluate
from .ground_truth import labelled_videos, load_label, load_or_create_split
from .media import probe
from .pipeline import load_stage_outputs

log = logging.getLogger(__name__)

TOLERANCES = (1.0, 2.0, 3.0)
OBJECTIVE_TOL = "2"  # tuning objective: macro boundary F1 at ±2 s

METHODS = ("baseline_all_shots", "baseline_adjacent", "sceneseen")


class NoLabelsError(RuntimeError):
    pass


def load_dataset(cfg: Config, split: str) -> list[dict]:
    """Labelled videos for a split, with cached shots/features loaded (computed if missing)."""
    items = labelled_videos(cfg.paths.ground_truth_dir, cfg.paths.videos_dir, cfg.paths.cache_dir)
    if not items:
        raise NoLabelsError(
            f"no labelled videos found: put videos in {cfg.paths.videos_dir} and labels in "
            f"{cfg.paths.ground_truth_dir} (see docs/EVALUATION.md)")
    sp = load_or_create_split(cfg.paths.ground_truth_dir, [s for s, *_ in items])
    wanted = set(sp["dev"]) | set(sp["test"]) if split == "all" else set(sp.get(split, []))
    data = []
    for stem, vpath, lpath in items:
        if stem not in wanted:
            continue
        info = probe(vpath)
        label = load_label(lpath, info.duration)
        t = time.perf_counter()
        stage = load_stage_outputs(cfg, info)
        data.append({"stem": stem, "info": info, "gt": label["boundaries"], "stage": stage,
                     "load_seconds": round(time.perf_counter() - t, 2)})
        log.info("loaded %s (%d shots, %d labelled boundaries)", stem, len(stage["shots"]), len(label["boundaries"]))
    if not data:
        raise NoLabelsError(f"split '{split}' has no videos (see {cfg.paths.ground_truth_dir}/splits.json)")
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
