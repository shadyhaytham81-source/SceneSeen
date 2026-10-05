"""Phase 2A orchestration:  scenes -> unique shots -> representative frames -> detector -> candidates.

Only the representative frame(s) of each UNIQUE shot are analysed; a camera set-up repeated
ten times costs one inference and its detections are credited to all ten occurrences.

Caching (all inside data/cache/<video_id>/commercial/, nothing machine-specific):

    frames_<size>/shot_XXXX_pYYY.jpg      representative frames (re-used by the UI for crops)
    detections_<detector key>.json        raw detections per FRAME CONTENT HASH

The detector key covers the model, its settings and the detector prompts. Relevance rules,
display thresholds, labels, icons and de-duplication are applied afterwards from the cached raw
detections, so changing them costs milliseconds and never re-runs the model, and nothing here
can invalidate Phase 1. If Phase 1 changes (new scenes / unique shots), only frames that were
not analysed before need inference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import CommercialConfig
from ..media import VideoError, VideoInfo
from ..shots import Shot
from . import frames as F
from . import taxonomy as T
from .attributes import dominant_color
from .dedup import frame_nms, max_instances, merge_shot_frames, scene_groups
from .detector import MODELS, DetectorUnavailable, get_detector
from .models import Candidate, Detection
from .relevance import commercial_relevance

log = logging.getLogger(__name__)
SCHEMA = 1
Progress = Callable[[str, float], None]


def detector_key(cfg: CommercialConfig) -> str:
    """Everything that changes what the detector would output for a given frame."""
    spec = MODELS.get(cfg.detector)
    src = [SCHEMA, cfg.detector, spec.model_id if spec else "?", spec.prompt_template if spec else "?",
           round(max(cfg.store_threshold, spec.native_threshold if spec else 0), 4), T.prompts_hash()]
    return hashlib.sha1(json.dumps(src).encode()).hexdigest()[:12]


class DetectionCache:
    """Raw detections per frame content hash. A corrupted file is set aside and rebuilt."""

    def __init__(self, directory: Path, key: str):
        self.path = directory / f"detections_{key}.json"
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict) or not all(isinstance(v, dict) and "dets" in v for v in loaded.values()):
                    raise ValueError("unexpected structure")
                self.data = loaded
            except (ValueError, OSError) as e:
                log.warning("commercial cache %s is unreadable (%s); rebuilding it", self.path.name, e)
                try:
                    self.path.replace(self.path.with_suffix(".corrupt"))
                except OSError:
                    pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)


def _frame_hashes(frames_dir: Path, paths: dict) -> dict:
    """{(shot, pos): content hash}, memoised in an index keyed by file name + size + mtime."""
    idx_path = frames_dir / "hashes.json"
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    except ValueError:
        idx = {}
    out, dirty = {}, False
    for k, p in paths.items():
        st = p.stat()
        sig = [st.st_size, int(st.st_mtime)]
        rec = idx.get(p.name)
        if not rec or rec[:2] != sig:
            rec = [*sig, F.image_hash(p)]
            idx[p.name] = rec
            dirty = True
        out[k] = rec[2]
    if dirty:
        idx_path.write_text(json.dumps(idx), encoding="utf-8")
    return out


def representative_shots(unique: dict) -> list[int]:
    return sorted({g["representative_shot_id"] for sc in unique["scenes"] for g in sc["unique"]})


def status_only(cache_dir: Path, cfg: CommercialConfig, unique: dict) -> dict:
    """Cheap check used by the UI: how much of this video is already analysed?"""
    cdir = cache_dir / "commercial"
    frames_dir = cdir / f"frames_{cfg.frame_long_side}"
    reps = representative_shots(unique)
    paths = {(s, p): frames_dir / F.frame_name(s, p) for s in reps for p in cfg.frame_positions}
    have = {k: p for k, p in paths.items() if p.exists()}
    cache = DetectionCache(cdir, detector_key(cfg))
    done = sum(1 for h in _frame_hashes(frames_dir, have).values() if h in cache.data) if have else 0
    return {"frames_needed": len(paths), "frames_analysed": done, "complete": done == len(paths) and len(paths) > 0}


def analyze(cache_dir: Path, info: VideoInfo, shots: list[Shot], scenes: list[dict], unique: dict,
            cfg: CommercialConfig, clip: np.ndarray | None = None, clip_model: tuple[str, str] | None = None,
            cache_root: Path | None = None, detector=None, allow_inference: bool = True,
            progress: Progress | None = None) -> dict:
    """Commercial analysis of one video. Never raises for model problems: returns
    status = "ready" | "partial" | "not_run" | "unavailable" with a human-readable reason."""
    t0 = time.perf_counter()
    progress = progress or (lambda stage, frac: None)
    timings: dict[str, float] = {}
    cdir = cache_dir / "commercial"
    frames_dir = cdir / f"frames_{cfg.frame_long_side}"
    reps = representative_shots(unique)
    positions = tuple(cfg.frame_positions)
    notes: list[str] = []
    status, reason = "ready", None

    # 1. representative frames (extracted once, cached)
    t = time.perf_counter()
    progress("frames", 0.0)
    try:
        paths = F.ensure_frames(info.path, shots, reps, positions, info.fps, cfg.frame_long_side, frames_dir) \
            if allow_inference else {(s, p): frames_dir / F.frame_name(s, p) for s in reps for p in positions
                                     if (frames_dir / F.frame_name(s, p)).exists()}
    except VideoError as e:
        paths = {(s, p): frames_dir / F.frame_name(s, p) for s in reps for p in positions
                 if (frames_dir / F.frame_name(s, p)).exists()}
        notes.append(str(e))
    hashes = _frame_hashes(frames_dir, paths) if paths else {}
    timings["frames"] = round(time.perf_counter() - t, 3)

    # 2. detections: cache first, model only for frames never seen with this detector key
    key = detector_key(cfg)
    cache = DetectionCache(cdir, key)
    todo = [k for k in paths if hashes[k] not in cache.data]
    spec = MODELS.get(cfg.detector)
    model_info = {"name": spec.model_id if spec else cfg.detector, "family": spec.family if spec else None,
                  "license": spec.license if spec else None, "detector_key": key}
    inferred, skipped = 0, 0
    t = time.perf_counter()
    if todo and allow_inference:
        try:
            det = detector or get_detector(cfg.detector, cfg.device, cfg.store_threshold, cfg.batch_size)
            prompts = [p for p, _ in T.detector_queries()]
            types = [tid for _, tid in T.detector_queries()]
            for i in range(0, len(todo), cfg.batch_size):
                chunk, images = [], []
                for k in todo[i:i + cfg.batch_size]:
                    try:
                        images.append(F.load_rgb(paths[k]))
                        chunk.append(k)
                    except ValueError as e:   # unreadable / unsupported frame: skip it, keep going
                        skipped += 1
                        log.warning("%s", e)
                if not images:
                    continue
                tb = time.perf_counter()
                results = det.detect(images, prompts)
                per = (time.perf_counter() - tb) / len(images)
                for k, res in zip(chunk, results):
                    cache.data[hashes[k]] = {"seconds": round(per, 4), "dets": [
                        {"type_id": types[qi], "prompt": prompts[qi], "score": round(s, 4),
                         "box": [round(v, 4) for v in box]} for qi, s, box in res]}
                inferred += len(chunk)
                cache.save()
                progress("detect", min(1.0, (i + cfg.batch_size) / len(todo)))
            model_info.update(det.describe() if hasattr(det, "describe") else {})
        except DetectorUnavailable as e:
            status, reason = ("partial" if any(hashes[k] in cache.data for k in paths) else "unavailable"), str(e)
            log.warning("commercial analysis degraded: %s", e)
    timings["inference"] = round(time.perf_counter() - t, 3)

    # status: how much of what this video needs has actually been analysed?
    needed = len(reps) * len(positions)
    analysed = sum(1 for k in paths if hashes[k] in cache.data)
    if status != "unavailable":
        if analysed >= needed:
            status = "ready"
        elif analysed == 0 and not allow_inference:
            status, reason = "not_run", "commercial analysis has not been run for this video yet"
        elif analysed == 0:
            status = "unavailable"
            reason = reason or (notes[0] if notes else "no representative frame could be analysed")
        else:
            status = "partial"
            reason = reason or f"{needed - analysed} of {needed} unique shots are not analysed yet"

    # 3. frame -> shot detections (cheap; recomputed from cached raw detections every time)
    t = time.perf_counter()
    dets_by_shot: dict[int, list[Detection]] = {}
    seconds = []
    for sid in reps:
        per_frame = []
        for p in positions:
            rec = cache.data.get(hashes.get((sid, p), ""))
            if rec is None:
                continue
            seconds.append(rec.get("seconds", 0.0))
            ds = [Detection(d["type_id"], d["prompt"], d["score"], d["box"], sid, p, hashes[(sid, p)])
                  for d in rec["dets"] if d["type_id"] in T.OBJECTS]
            per_frame.append(frame_nms(ds))
        if per_frame:
            dets_by_shot[sid] = per_frame[0] if len(per_frame) == 1 else merge_shot_frames(per_frame)

    # 4. scene context from Phase 1's cached CLIP embeddings
    contexts: list[dict] | None = None
    if clip is not None and clip_model and cache_root is not None:
        try:
            from .scene_context import classify_scenes, text_embeddings

            contexts = classify_scenes(clip, shots, scenes, text_embeddings(cache_root, *clip_model),
                                       cfg.context_min_confidence, cfg.context_min_margin)
        except Exception as e:   # CLIP unavailable: objects still work
            notes.append(f"scene context unavailable: {type(e).__name__}: {e}")
            log.warning("scene context unavailable: %s", e)

    # 5. scene-level candidates
    uniq_by_scene = {sc["scene_id"]: sc for sc in unique["scenes"]}
    out_scenes, n_shown, n_hidden = [], 0, 0
    color_cache: dict[str, np.ndarray] = {}
    for i, sc in enumerate(scenes):
        usc = uniq_by_scene.get(sc["scene_id"])
        cands: list[Candidate] = []
        analysed = 0
        if usc:
            analysed = sum(1 for g in usc["unique"] if g["representative_shot_id"] in dets_by_shot)
            for tid, grp in scene_groups(usc["unique"], dets_by_shot).items():
                obj = T.OBJECTS[tid]
                best = max(grp["detections"], key=lambda d: (d.score, d.area))
                occ = grp["occurrences"]
                rel, factors = commercial_relevance(obj, best.box, len(occ))
                need = max(cfg.min_confidence, obj.min_confidence or 0.0)
                hidden = ("low detector confidence" if best.score < need
                          else "low commercial relevance" if rel < cfg.min_relevance else None)
                color = None
                if cfg.describe_colors and obj.describe_color and hidden is None:
                    color = _color_of(best, paths, color_cache)
                cat = T.CATEGORIES[obj.category]
                ckey = hashlib.sha1(f"{tid}|{best.frame_hash}|{[round(v, 3) for v in best.box]}".encode()).hexdigest()[:12]
                cands.append(Candidate(
                    key=ckey, scene_id=sc["scene_id"], type_id=tid,
                    label=f"{color.capitalize()} {obj.label.lower()}" if color else obj.label,
                    category=obj.category, category_name=cat.name, icon=obj.icon or cat.icon,
                    detection_confidence=round(best.score, 4), commercial_relevance=rel, displayed=hidden is None,
                    hidden_reason=hidden, seen_count=len(occ), detected_in_unique_shots=len(grp["unique_shots"]),
                    first_seen=occ[0]["start"], last_seen=occ[-1]["end"],
                    screen_time=round(sum(o["end"] - o["start"] for o in occ), 3),
                    max_instances=max_instances(grp["detections"]), color=color,
                    best={"shot_id": best.shot_id, "position": best.position, "box": best.box,
                          "frame_hash": best.frame_hash},
                    occurrences=occ,
                    debug={"raw_label": best.prompt, "relevance_factors": factors, "threshold": need,
                           "unique_shot_ids": grp["unique_shots"],
                           "detections": [d.to_dict() | {"shot_id": d.shot_id} for d in
                                          sorted(grp["detections"], key=lambda d: -d.score)[:12]]}))
        cands.sort(key=lambda c: (not c.displayed, -c.commercial_relevance * c.detection_confidence))
        shown = [c for c in cands if c.displayed]
        n_shown += len(shown)
        n_hidden += len(cands) - len(shown)
        ctx = contexts[i] if contexts else {"venue": None, "environment": None, "ranked": []}
        cats = sorted({c.category for c in shown} | ({ctx["venue"]["category"]} if ctx["venue"] and ctx["venue"]["category"] else set()))
        out_scenes.append({
            "scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
            "context": ctx, "categories_present": [{"id": c, "name": T.CATEGORIES[c].name} for c in cats],
            "unique_shots": len(usc["unique"]) if usc else 0, "unique_shots_analysed": analysed,
            "candidates": [c.to_dict() for c in cands]})
    timings["assemble"] = round(time.perf_counter() - t, 3)
    timings["total"] = round(time.perf_counter() - t0, 3)
    n_frames = len(reps) * len(positions)
    return {
        "status": status, "reason": reason, "notes": notes,
        "taxonomy_version": T.TAXONOMY_VERSION, "model": model_info, "config": asdict(cfg),
        "categories": [{"id": c.id, "name": c.name, "icon": c.icon} for c in T.CATEGORIES.values()],
        "summary": {"scenes": len(out_scenes), "original_shots": len(shots), "unique_shots": len(reps),
                    "frames_needed": n_frames, "frames_analysed": sum(1 for k in paths if hashes[k] in cache.data),
                    "frames_inferred_now": inferred, "frames_from_cache": max(0, len(paths) - len(todo)),
                    "frames_skipped": skipped, "candidates_shown": n_shown, "candidates_hidden": n_hidden,
                    "inference_avoided_by_unique_shots": len(shots) - len(reps),
                    "mean_seconds_per_frame": round(float(np.mean(seconds)), 4) if seconds else None},
        "timings": timings, "scenes": out_scenes,
    }


def _color_of(d: Detection, paths: dict, cache: dict) -> str | None:
    p = paths.get((d.shot_id, d.position))
    if p is None:
        return None
    try:
        if d.frame_hash not in cache:
            cache.clear()               # keep at most one decoded frame in memory
            cache[d.frame_hash] = F.load_rgb(p)
        return dominant_color(cache[d.frame_hash], d.box)
    except ValueError:
        return None
