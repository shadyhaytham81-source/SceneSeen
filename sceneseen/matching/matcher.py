"""Product matching: detected commercial object -> ranked catalogue candidates -> honest state.

Four separate things are kept separate all the way to the UI (never merged into one number):

    detection_confidence   how sure the detector is that the object is there / correctly typed
    commercial_relevance   how commercially interesting that kind of object is in this scene
    match_confidence       how likely the BEST catalogue candidate is the same product
    verification_status    what a human decided (verification.py)

Matching evidence (each one measured in scripts/product_matching_study.py, see docs/MATCHING.md):

    category gating        only products of a compatible object type are compared at all
    embedding cosine       OpenCLIP image embedding of the object crop vs. reference images
    colour                 HSV-histogram intersection (the strongest extra signal in the study)
    several references     a product scores with its best reference image
    several occurrences    quality-weighted mean over the object's occurrence crops
    crop size              small crops count less

The nearest product is NOT automatically "the" product: below the calibrated thresholds the
answer is "unknown product".
"""
from __future__ import annotations

import logging
import math
import time
from typing import Callable

import numpy as np

from ..commercial import frames as F
from ..config import MatchingConfig
from .embedder import EmbedderUnavailable
from .index import CatalogIndex
from .signals import colour_signature, crop_weight
from .store import CropCache

log = logging.getLogger(__name__)

STATES = ("high_confidence", "possible_match", "uncertain", "no_reliable_match")
STATE_LABELS = {"high_confidence": "High confidence", "possible_match": "Possible match", "uncertain": "Uncertain",
                "no_reliable_match": "No reliable match"}


def match_confidence(score: float, margin: float, cfg: MatchingConfig) -> float:
    """Calibrated probability (0-1) that a candidate is the same product, from its score and its
    lead over the best OTHER product. In the calibration study the lead predicted correctness far
    better (AUC 0.87) than the raw score (0.72): a lookalike rarely stands out from the rest."""
    z = cfg.conf_w_score * score + cfg.conf_w_margin * margin + cfg.conf_bias
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))


def match_state(confidence: float, cfg: MatchingConfig, cap_at_possible: bool = False) -> str:
    if confidence >= cfg.high_confidence and not cap_at_possible:
        return "high_confidence"
    if confidence >= cfg.possible_confidence:
        return "possible_match"
    if confidence >= cfg.uncertain_confidence:
        return "uncertain"
    return "no_reliable_match"


def select_occurrences(candidate: dict, limit: int) -> list[dict]:
    """The object's best detections (by confidence and size), one per shot, as matching evidence."""
    seen, out = set(), []
    dets = sorted(candidate.get("debug", {}).get("detections", []),
                  key=lambda d: -(d["score"] * math.sqrt(max(1e-6, (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])))))
    for d in dets:
        if d["shot_id"] in seen:
            continue
        seen.add(d["shot_id"])
        out.append(d)
        if len(out) >= limit:
            break
    return out


def crop_rgb(frame: np.ndarray, box: list[float], pad: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    a, b = int(max(0, (x1 - px) * w)), int(max(0, (y1 - py) * h))
    c, d = int(min(w, (x2 + px) * w)), int(min(h, (y2 + py) * h))
    return np.ascontiguousarray(frame[b:max(d, b + 1), a:max(c, a + 1)])


def rank(index: CatalogIndex, emb: np.ndarray, col: np.ndarray, sides: np.ndarray, type_id: str,
         cfg: MatchingConfig) -> dict:
    """Ranked candidates + state for one object, from its occurrence embeddings."""
    weights = np.array([crop_weight(s) for s in sides])
    res = index.search(emb, col, weights, type_id, cfg.colour_weight, cfg.top_k, cfg.shortlist)
    cands = res["results"]
    if not cands:
        reason = "no catalogue products for this kind of object" if res["eligible_products"] == 0 else "no candidates"
        return {"state": "no_reliable_match", "reason": reason, "eligible_products": res["eligible_products"],
                "match_confidence": 0.0, "candidates": [], "occurrences_used": int(len(emb))}
    # Lead of each candidate over the best OTHER product. With very few comparable products
    # there is no meaningful runner-up, so it is floored at the score a wrong product typically
    # reaches, and the result can be "possible" at best.
    small = res["eligible_products"] < cfg.min_products_for_high
    scores = [c["score"] for c in cands]
    for i, c in enumerate(cands):
        other = max([s for j, s in enumerate(scores) if j != i], default=-1.0)
        if small:
            other = max(other, cfg.background_score)
        c["rank"] = i + 1
        c["margin"] = round(c["score"] - other, 4)
        c["match_confidence"] = round(match_confidence(c["score"], c["score"] - other, cfg), 4)
        c["score"], c["cosine"], c["colour"] = round(c["score"], 4), round(c["cosine"], 4), round(c["colour"], 4)
    top = cands[0]
    state = match_state(top["match_confidence"], cfg, cap_at_possible=small)
    note = (f"only {res['eligible_products']} comparable product(s) in the catalogue: "
            "high confidence needs at least " f"{cfg.min_products_for_high}") if small else None
    return {"state": state, "reason": note, "eligible_products": res["eligible_products"],
            "match_confidence": top["match_confidence"], "margin": top["margin"], "margin_estimated": small,
            "candidates": cands, "occurrences_used": int(len(emb)),
            "mean_crop_side": round(float(np.mean(sides)), 1)}


def match_video(commercial: dict, commercial_dir, frames_dir, index: CatalogIndex, embedder, cfg: MatchingConfig,
                allow_inference: bool = True, progress: Callable[[str, float], None] | None = None) -> dict:
    """Match every DISPLAYED commercial object of a video against the catalogue.

    Returns {"status": ready | partial | not_run | unavailable, "matches": {candidate key: ...}}.
    Never raises for model problems; commercial objects stay usable without matches.
    """
    t0 = time.perf_counter()
    progress = progress or (lambda stage, frac: None)
    key = embedder.cache_key()
    cache = CropCache(commercial_dir, key)
    cands = [c for sc in commercial.get("scenes", []) for c in sc["candidates"] if c["displayed"]]
    plan, need = {}, {}
    for c in cands:
        occ = select_occurrences(c, cfg.max_occurrences) or [{"shot_id": c["best"]["shot_id"], "box": c["best"]["box"],
                                                              "score": c["detection_confidence"]}]
        items = []
        for d in occ:
            fp = frames_dir / F.frame_name(d["shot_id"], 0.5)
            items.append((d, fp))
        plan[c["key"]] = items
    # which crops still need embedding?
    hashes: dict = {}
    for ckey, items in plan.items():
        for d, fp in items:
            if not fp.exists():
                continue
            fh = hashes.setdefault(fp, F.image_hash(fp))
            k = CropCache.key(fh, d["box"])
            if cache.get(k) is None:
                need[k] = (fp, d["box"])
    status, reason, embedded = "ready", None, 0
    if need and allow_inference:
        try:
            keys = list(need)
            embs, cols, sides, done = [], [], [], []
            frame_cache: dict = {}
            for i in range(0, len(keys), cfg.batch_size):
                chunk, crops = [], []
                for k in keys[i:i + cfg.batch_size]:
                    fp, box = need[k]
                    try:
                        if fp not in frame_cache:
                            if len(frame_cache) > 8:
                                frame_cache.clear()
                            frame_cache[fp] = F.load_rgb(fp)
                        crop = crop_rgb(frame_cache[fp], box, cfg.crop_pad)
                    except ValueError as e:
                        log.warning("%s", e)
                        continue
                    crops.append(crop)
                    chunk.append(k)
                if not crops:
                    continue
                embs.append(embedder.embed(crops))
                cols.append(np.stack([colour_signature(c) for c in crops]))
                sides.append(np.array([min(c.shape[0], c.shape[1]) for c in crops], np.float32))
                done += chunk
                progress("embed_objects", min(1.0, (i + cfg.batch_size) / len(keys)))
            if done:
                cache.add(done, np.concatenate(embs), np.concatenate(cols), np.concatenate(sides))
                embedded = len(done)
        except EmbedderUnavailable as e:
            status, reason = "unavailable", str(e)
            log.warning("product matching degraded: %s", e)
    elif need:
        status, reason = "not_run", f"{len(need)} object crop(s) have not been embedded yet"
    matches, missing = {}, 0
    for c in cands:
        embs, cols, sides = [], [], []
        for d, fp in plan[c["key"]]:
            if fp not in hashes:
                continue
            got = cache.get(CropCache.key(hashes[fp], d["box"]))
            if got is not None:
                embs.append(got[0])
                cols.append(got[1])
                sides.append(got[2])
        if not embs:
            missing += 1
            continue
        try:
            matches[c["key"]] = rank(index, np.stack(embs), np.stack(cols), np.array(sides), c["type_id"], cfg)
        except Exception as e:   # one bad object must not lose the others
            log.warning("matching failed for %s: %s", c["key"], e)
            missing += 1
    progress("match", 1.0)
    if status == "ready" and missing:
        status = "partial" if matches else ("not_run" if not allow_inference else "unavailable")
        reason = reason or f"{missing} object(s) could not be matched"
    if status in ("not_run", "unavailable") and matches:
        status = "partial"
    return {"status": status, "reason": reason, "matches": matches,
            "model": {"embedder_key": key, **(embedder.describe() if embedder.device else {})},
            "calibration_version": cfg.calibration_version,
            "summary": {"objects": len(cands), "matched_against_catalogue": len(matches),
                        "crops_embedded_now": embedded, "catalogue_products": index.n_products,
                        "catalogue_images": len(index),
                        "states": {s: sum(1 for m in matches.values() if m["state"] == s) for s in STATES}},
            "seconds": round(time.perf_counter() - t0, 3)}
