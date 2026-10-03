"""End-to-end orchestration with caching.

Expensive stages (shot detection, frame sampling + embeddings) are cached per
video under data/cache/<video_id>/, keyed by a hash of the config section that
produced them. Re-grouping with new thresholds only re-runs grouping.py
(milliseconds).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import grouping
from .config import Config, GroupingConfig, config_hash
from .features import extract_features
from .media import VideoInfo, probe
from .shots import detect_shots, shots_from_record

log = logging.getLogger(__name__)

# stage name -> user-facing label (the UI never shows ML jargon)
STAGES = {
    "reading": "Reading video",
    "shots": "Detecting shots",
    "scenes": "Understanding scenes",
    "results": "Preparing results",
}
Progress = Callable[[str, float], None]


def _noop(stage: str, frac: float) -> None:
    pass


class VideoCache:
    def __init__(self, cache_root: Path, video_id: str):
        self.dir = cache_root / video_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.dir / name

    def read_json(self, name: str):
        p = self.path(name)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def write_json(self, name: str, data) -> None:
        tmp = self.path(name + ".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path(name))


def _keys(cfg: Config, info: VideoInfo) -> tuple[str, str]:
    shots_key = config_hash(cfg.shots)
    feat_key = f"{shots_key}_{config_hash(cfg.features)}"
    return shots_key, feat_key


def load_stage_outputs(cfg: Config, info: VideoInfo, progress: Progress = _noop) -> dict:
    """Shots + features for a video, computing (and caching) whatever is missing."""
    cache = VideoCache(cfg.paths.cache_dir, info.video_id)
    cache.write_json("info.json", info.to_dict())
    shots_key, feat_key = _keys(cfg, info)
    timings: dict[str, float] = {}

    progress("shots", 0.0)
    shots_name = f"shots_{shots_key}.json"
    shots_rec = cache.read_json(shots_name)
    if shots_rec is None:
        t = time.perf_counter()
        shots_rec = detect_shots(info, cfg.shots)
        shots_rec["seconds"] = round(time.perf_counter() - t, 2)
        cache.write_json(shots_name, shots_rec)
        timings["shot_detection"] = shots_rec["seconds"]
    else:
        timings["shot_detection_cached"] = shots_rec.get("seconds", 0.0)
    shots = shots_from_record(shots_rec)
    progress("shots", 1.0)

    progress("scenes", 0.0)
    feat_path = cache.path(f"features_{feat_key}.npz")
    thumbs_dir = cache.path(f"thumbs_{feat_key}")
    if feat_path.exists():
        feats = dict(np.load(feat_path))
        timings["features_cached"] = float(feats.get("seconds", 0.0))
    else:
        t = time.perf_counter()
        feats = extract_features(info, shots, cfg.features, thumbs_dir, lambda f: progress("scenes", 0.9 * f))
        feats["seconds"] = np.float32(time.perf_counter() - t)
        np.savez_compressed(feat_path, **feats)
        timings["features"] = round(float(feats["seconds"]), 2)
    return {"shots_rec": shots_rec, "shots": shots, "features": feats, "timings": timings,
            "thumbs_dir": thumbs_dir, "feat_key": feat_key, "cache": cache}


def apply_boundary_model(model, info: VideoInfo, stage: dict, gcfg: GroupingConfig, g: dict) -> dict:
    """Replace the rule's boundary decisions with a trained classifier's (cut scores are kept for
    the developer view). Only called when a model is explicitly configured and loaded."""
    from .learning import boundary_features

    shots = stage["shots"]
    X = boundary_features(shots, stage["features"]["clip"], stage["features"]["color"], gcfg,
                          stage["shots_rec"].get("transition_probs"), info.fps)
    times = np.array([s.start for s in shots[1:]])
    prob = model.predict_proba(X) if len(X) else np.zeros(0)
    bounds = model.boundaries(times, X, info.duration)
    chosen = set(bounds)
    for c, p in zip(g["cuts"], prob):
        c["rule_decision"] = c["decision"]
        c["probability"] = round(float(p), 4)
        c["decision"] = "boundary" if round(c["time"], 3) in chosen else (
            "min_duration" if p >= model.threshold else "below_abs")
    return {**g, "boundaries": bounds, "scenes": grouping.boundaries_to_scenes(bounds, info.duration, shots)}


def build_result(info: VideoInfo, stage: dict, gcfg: GroupingConfig, timings: dict, model=None) -> dict:
    """`model`: optional learning.BoundaryModel. None (default) = hand-designed rule."""
    t = time.perf_counter()
    shots = stage["shots"]
    g = grouping.group_shots(shots, stage["features"]["clip"], stage["features"]["color"], info.duration, gcfg)
    if model is not None and len(shots) > 1:
        g = apply_boundary_model(model, info, stage, gcfg, g)
    timings = {**timings, "grouping": round(time.perf_counter() - t, 4)}
    return {
        "video": Path(info.path).name,
        "video_id": info.video_id,
        "duration": info.duration,
        "fps": info.fps,
        "scene_count": len(g["scenes"]),
        "scenes": g["scenes"],
        "boundaries": g["boundaries"],
        "shot_count": len(shots),
        "debug": {
            "shot_detector": stage["shots_rec"]["detector"],
            "raw_shot_count": stage["shots_rec"]["raw_shot_count"],
            "shots": [s.to_dict() for s in shots],
            "cuts": g["cuts"],
            "adjacent_similarity": g.get("adjacent_similarity", []),
            "grouping_config": asdict(gcfg),
            "timings": timings,
            "feature_key": stage["feat_key"],
            "baseline_all_shots_count": len(shots),
            "boundary_method": "rule" if model is None else f"learned:{model.to_dict()['artifact_hash']}",
        },
    }


def analyze(path: str | Path, cfg: Config, progress: Progress = _noop, gcfg: GroupingConfig | None = None) -> dict:
    t0 = time.perf_counter()
    progress("reading", 0.0)
    info = probe(path)
    log.info("analyzing %s (%.1fs, %.3f fps, %dx%d) id=%s", Path(path).name, info.duration, info.fps,
             info.width, info.height, info.video_id)
    progress("reading", 1.0)
    stage = load_stage_outputs(cfg, info, progress)
    progress("results", 0.0)
    from .config import ROOT
    from .learning import load_model_or_none

    model = load_model_or_none(cfg.boundary_model.path, ROOT)   # None unless explicitly configured
    result = build_result(info, stage, gcfg or cfg.grouping, stage["timings"], model)
    result["debug"]["timings"]["total"] = round(time.perf_counter() - t0, 2)
    stage["cache"].write_json("result.json", result)
    progress("results", 1.0)
    log.info("%s: %d shots -> %d scenes in %.1fs", info.video_id, result["shot_count"], result["scene_count"],
             result["debug"]["timings"]["total"])
    return result


def public_result(result: dict) -> dict:
    """Scene metadata export (no debug internals)."""
    keys = ("video", "duration", "scene_count", "scenes")
    out = {k: result[k] for k in keys}
    out["scenes"] = [{k: s[k] for k in ("scene_id", "start_seconds", "end_seconds", "duration_seconds")}
                     for s in result["scenes"]]
    if result.get("corrected"):
        out["corrected"] = True
    return out


def unique_shots_for(cfg: Config, info: VideoInfo, stage: dict, scenes: list[dict]) -> dict:
    """Repeated-shot grouping for the given scenes (downstream of segmentation; cached)."""
    from .unique import unique_shots_cached

    return unique_shots_cached(stage["cache"].dir, stage["feat_key"], stage["thumbs_dir"], stage["shots"], scenes,
                               stage["features"], cfg.unique_shots)
