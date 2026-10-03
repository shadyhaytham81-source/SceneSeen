"""Unique Shots: group visually repeated shots (same camera set-up) inside each scene.

This is pure post-processing. It reads the shots, the cached CLIP embeddings / colour
histograms / thumbnails and the scene boundaries, and it never changes any of them: every
original shot is kept and linked to exactly one unique-shot group.

Similarity (calibrated in scripts/unique_shots_study.py on 177 labelled shot pairs):

    p(duplicate) = sigmoid( w_clip * clip_matched + w_color * color + w_phash * phash + bias )

* clip_matched  cosine similarity of L2-normalised OpenCLIP embeddings, matched across the
                sampled frames of both shots (symmetric best-match average)  -> "same content"
* color         HSV histogram intersection                                   -> "same light / palette"
* phash         perceptual-hash agreement of the mid-shot frame             -> "same framing / layout"

CLIP alone cannot tell "same actor, different angle" from "same camera set-up"; the layout
term can. Pairs are Duplicate (p >= duplicate_threshold), Different (p < different_threshold)
or Uncertain (in between; reported, never merged). Groups are formed with average-linkage
clustering so that one borderline link cannot chain unrelated shots together.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import similarity as sim
from .config import UniqueShotsConfig
from .shots import Shot

log = logging.getLogger(__name__)
VERSION = 1


# ---------------------------------------------------------------- per-shot visual stats (cached)

def thumbnail_stats(thumbs_dir: Path, n_shots: int) -> dict[str, np.ndarray]:
    """Perceptual hash, sharpness and brightness of each shot's mid-frame thumbnail."""
    import cv2

    bits = np.zeros((n_shots, 64), bool)
    sharp = np.zeros(n_shots, np.float32)
    bright = np.full(n_shots, 0.5, np.float32)
    have = np.zeros(n_shots, bool)
    for i in range(n_shots):
        p = thumbs_dir / f"shot_{i:04d}.jpg"
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        if img is None:
            continue
        small = cv2.resize(img, (128, 72), interpolation=cv2.INTER_AREA)
        bits[i] = sim.phash_bits(small)
        sharp[i] = float(cv2.Laplacian(img, cv2.CV_64F).var())
        bright[i] = float(img.mean()) / 255.0
        have[i] = True
    return {"phash": bits, "sharpness": sharp, "brightness": bright, "has_thumb": have}


def load_thumbnail_stats(cache_dir: Path, feat_key: str, thumbs_dir: Path, n_shots: int) -> dict[str, np.ndarray]:
    path = cache_dir / f"visual_{feat_key}.npz"
    if path.exists():
        d = dict(np.load(path))
        if len(d["sharpness"]) == n_shots:
            return d
    d = thumbnail_stats(thumbs_dir, n_shots)
    if d["has_thumb"].any():
        np.savez_compressed(path, **d)
    return d


# ---------------------------------------------------------------- similarity

def duplicate_probability(clip: np.ndarray, color: np.ndarray, phash: np.ndarray, cfg: UniqueShotsConfig,
                          has_thumb: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """[N, N] matrices: clip_matched, color, phash and the combined duplicate probability."""
    fs = sim.clip_frame_similarity(clip)
    c = sim.clip_matched_similarity(fs)
    col = sim.color_similarity(color)
    ph = sim.hash_similarity(phash)
    if has_thumb is not None and not has_thumb.all():
        # no thumbnail -> no layout evidence: fall back to a neutral value for those pairs
        missing = ~(has_thumb[:, None] & has_thumb[None, :])
        ph = np.where(missing, cfg.phash_neutral, ph)
    z = cfg.w_clip * c + cfg.w_color * col + cfg.w_phash * ph + cfg.bias
    return {"clip": c, "color": col, "phash": ph, "prob": 1.0 / (1.0 + np.exp(-z))}


def cluster(prob: np.ndarray, threshold: float, linkage: str = "average") -> np.ndarray:
    """Group labels (0..k-1, in order of first appearance) for one scene's shots."""
    n = len(prob)
    if n == 1:
        return np.zeros(1, int)
    from scipy.cluster.hierarchy import fcluster, linkage as _linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - (prob + prob.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    raw = fcluster(_linkage(squareform(dist, checks=False), method=linkage), t=1.0 - threshold, criterion="distance")
    order: dict[int, int] = {}
    return np.array([order.setdefault(int(r), len(order)) for r in raw])


# ---------------------------------------------------------------- representative

def _rank01(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1] (ties share the average rank); a single element gets 1."""
    if len(x) == 1:
        return np.ones(1)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    for v in np.unique(x):
        m = x == v
        order[m] = order[m].mean()
    return order / (len(x) - 1)


def representative_scores(idx: np.ndarray, prob: np.ndarray, shots: list[Shot], clip: np.ndarray,
                          stats: dict[str, np.ndarray], cfg: UniqueShotsConfig) -> np.ndarray:
    """Score each member of a group; the highest becomes the representative.

    centrality   typical of the group (mean duplicate probability to the other members)
    sharpness    Laplacian variance of the mid-frame (not blurred / not mid-motion)
    duration     longer occurrences show more (log seconds)
    stability    frames inside the shot agree with each other (low = fade, whip-pan, transition)
    exposure     mid-grey brightness preferred over near-black / blown-out frames
    """
    sub = prob[np.ix_(idx, idx)]
    centrality = (sub.sum(axis=1) - np.diag(sub)) / max(1, len(idx) - 1)
    sharp = np.log1p(stats["sharpness"][idx])
    dur = np.log1p(np.array([shots[i].end - shots[i].start for i in idx]))
    e = sim.normalize(clip[idx].astype(np.float32))
    stability = np.einsum("nad,nbd->nab", e, e).min(axis=(1, 2))
    exposure = 1.0 - np.abs(stats["brightness"][idx] - 0.45) / 0.55
    return (cfg.rep_w_centrality * _rank01(centrality) + cfg.rep_w_sharpness * _rank01(sharp)
            + cfg.rep_w_duration * _rank01(dur) + cfg.rep_w_stability * _rank01(stability)
            + cfg.rep_w_exposure * _rank01(exposure))


# ---------------------------------------------------------------- main

def unique_shots(shots: list[Shot], scenes: list[dict], clip: np.ndarray, color: np.ndarray,
                 stats: dict[str, np.ndarray], cfg: UniqueShotsConfig) -> dict:
    """Group repeated shots per scene. Returns a JSON-serialisable record."""
    t0 = time.perf_counter()
    out_scenes = []
    total_unique = 0
    for sc in scenes:
        if sc.get("shot_start") is None:
            continue
        ids = np.arange(sc["shot_start"], sc["shot_end"] + 1)
        m = duplicate_probability(clip[ids], color[ids], stats["phash"][ids], cfg, stats["has_thumb"][ids])
        prob = m["prob"]
        labels = cluster(prob, cfg.duplicate_threshold, cfg.linkage)
        groups = []
        for g in range(int(labels.max()) + 1):
            local = np.where(labels == g)[0]
            scores = representative_scores(local, prob, [shots[i] for i in ids], clip[ids],
                                           {k: v[ids] for k, v in stats.items()}, cfg)
            rep_local = int(local[int(np.argmax(scores))])
            occ = [{
                "shot_id": int(ids[j]),
                "start": shots[ids[j]].start,
                "end": shots[ids[j]].end,
                "similarity_to_representative": round(float(prob[j, rep_local]) if j != rep_local else 1.0, 4),
                "clip_cosine_to_representative": round(float(m["clip"][j, rep_local]), 4),
                "representative_score": round(float(scores[k]), 4),
            } for k, j in enumerate(local)]
            groups.append({
                "unique_shot_id": f"S{sc['scene_id']:02d}-U{g + 1:02d}",
                "scene_id": sc["scene_id"],
                "representative_shot_id": int(ids[rep_local]),
                "shot_ids": [int(ids[j]) for j in local],
                "duplicate_shot_ids": [int(ids[j]) for j in local if j != rep_local],
                "occurrence_count": len(local),
                "first_seen": shots[ids[local[0]]].start,
                "total_duration": round(float(sum(shots[ids[j]].end - shots[ids[j]].start for j in local)), 3),
                "occurrences": occ,
                "_local": local,
            })
        # uncertain relations between groups: reported for review, never merged
        for a in groups:
            rel = []
            for b in groups:
                if a is b:
                    continue
                best = float(prob[np.ix_(a["_local"], b["_local"])].max())
                if best >= cfg.different_threshold:
                    rel.append({"unique_shot_id": b["unique_shot_id"], "max_similarity": round(best, 4)})
            a["possible_duplicates_of"] = sorted(rel, key=lambda r: -r["max_similarity"])
        for g_ in groups:
            del g_["_local"]
        n, u = len(ids), len(groups)
        total_unique += u
        out_scenes.append({
            "scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
            "original_shots": n, "unique_shots": u, "repeated_shots": n - u,
            "reduction": round(1 - u / n, 4) if n else 0.0, "unique": groups,
        })
    n_all = len(shots)
    return {
        "version": VERSION,
        "method": "sigmoid(w_clip*clip_matched + w_color*color + w_phash*phash + bias); average-linkage per scene",
        "config": asdict(cfg),
        "summary": {"original_shots": n_all, "unique_shots": total_unique, "repeated_shots": n_all - total_unique,
                    "reduction": round(1 - total_unique / n_all, 4) if n_all else 0.0, "scenes": len(out_scenes)},
        "scenes": out_scenes,
        "seconds": round(time.perf_counter() - t0, 4),
    }


def unique_shots_cached(cache_dir: Path, feat_key: str, thumbs_dir: Path, shots: list[Shot], scenes: list[dict],
                        features: dict, cfg: UniqueShotsConfig) -> dict:
    """Unique shots for the given scenes, cached per (features, config, scene boundaries)."""
    key_src = json.dumps([asdict(cfg), [(s["shot_start"], s["shot_end"]) for s in scenes], VERSION], sort_keys=True)
    path = cache_dir / f"unique_{feat_key}_{hashlib.sha1(key_src.encode()).hexdigest()[:10]}.json"
    if path.exists():
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["cached"] = True
        return rec
    t0 = time.perf_counter()
    stats = load_thumbnail_stats(cache_dir, feat_key, thumbs_dir, len(shots))
    rec = unique_shots(shots, scenes, features["clip"], features["color"], stats, cfg)
    rec["seconds_total"] = round(time.perf_counter() - t0, 4)
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    rec["cached"] = False
    return rec
