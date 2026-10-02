"""Scene-boundary evaluation against human labels. Pure functions, no I/O.

Metrics
-------
* Boundary precision / recall / F1 at a time tolerance (±1s, ±2s, ±3s, ...).
  Predicted and true boundaries are matched one-to-one (optimal assignment on
  |time difference|), so one prediction can never satisfy two labels.
* Mean absolute timing error of the matched pairs.
* Count ratio = predicted / true scenes (>1 over-segmentation, <1 under-segmentation).
* Coverage / Overflow (Vendrig & Worring 2002; standard in scene-segmentation work,
  e.g. Rotman et al.): coverage drops when true scenes are split (over-segmentation),
  overflow rises when predicted scenes spill across true scenes (under-segmentation).
  F_CO = harmonic mean of coverage and (1 - overflow).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _clean(bounds, duration: float) -> np.ndarray:
    return np.array(sorted(b for b in set(float(x) for x in bounds) if 0.0 < b < duration))


def match_boundaries(pred, gt, tolerance: float) -> list[tuple[int, int, float]]:
    """Optimal one-to-one matching; returns (pred_idx, gt_idx, |error|) with error <= tolerance."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    if len(pred) == 0 or len(gt) == 0:
        return []
    cost = np.abs(pred[:, None] - gt[None, :])
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(cost <= tolerance, cost, big))
    return [(int(r), int(c), float(cost[r, c])) for r, c in zip(rows, cols) if cost[r, c] <= tolerance]


def boundary_prf(pred, gt, tolerance: float) -> dict:
    m = match_boundaries(pred, gt, tolerance)
    tp = len(m)
    p = tp / len(pred) if len(pred) else (1.0 if len(gt) == 0 else 0.0)
    r = tp / len(gt) if len(gt) else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "tolerance": tolerance,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": len(pred) - tp,
        "fn": len(gt) - tp,
        "mean_abs_error": float(np.mean([e for *_, e in m])) if m else None,
    }


def _segments(bounds: np.ndarray, duration: float) -> list[tuple[float, float]]:
    edges = [0.0, *bounds.tolist(), duration]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def coverage_overflow(pred, gt, duration: float) -> dict:
    P = _segments(np.asarray(pred, float), duration)
    G = _segments(np.asarray(gt, float), duration)
    total = sum(g[1] - g[0] for g in G)
    cov = sum(max(_overlap(g, p) for p in P) for g in G) / total
    ov_sum = 0.0
    for t, g in enumerate(G):
        glen = g[1] - g[0]
        neigh = (G[t - 1][1] - G[t - 1][0] if t > 0 else 0.0) + (G[t + 1][1] - G[t + 1][0] if t + 1 < len(G) else 0.0)
        if neigh <= 0:
            continue  # single-scene video: overflow undefined -> 0
        spill = sum((p[1] - p[0]) - _overlap(p, g) for p in P if _overlap(p, g) > 0)
        ov_sum += glen * min(1.0, spill / neigh)
    ov = ov_sum / total
    f = 2 * cov * (1 - ov) / (cov + 1 - ov) if cov + 1 - ov > 0 else 0.0
    return {"coverage": cov, "overflow": ov, "f_co": f}


def evaluate(pred, gt, duration: float, tolerances=(1.0, 2.0, 3.0)) -> dict:
    pred_b, gt_b = _clean(pred, duration), _clean(gt, duration)
    out = {
        "n_pred_boundaries": len(pred_b),
        "n_gt_boundaries": len(gt_b),
        "n_pred_scenes": len(pred_b) + 1,
        "n_gt_scenes": len(gt_b) + 1,
        "count_ratio": (len(pred_b) + 1) / (len(gt_b) + 1),
        "by_tolerance": {f"{t:g}": boundary_prf(pred_b, gt_b, t) for t in tolerances},
    }
    out.update(coverage_overflow(pred_b, gt_b, duration))
    return out


def aggregate(per_video: list[dict]) -> dict:
    """Macro-average (each video counts equally) + micro P/R/F1 (pooled counts)."""
    if not per_video:
        return {}
    tols = per_video[0]["by_tolerance"].keys()
    agg: dict = {"videos": len(per_video)}
    for k in ("coverage", "overflow", "f_co", "count_ratio"):
        agg[k] = float(np.mean([v[k] for v in per_video]))
    agg["by_tolerance"] = {}
    for t in tols:
        rows = [v["by_tolerance"][t] for v in per_video]
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        errs = [r_["mean_abs_error"] for r_ in rows if r_["mean_abs_error"] is not None]
        agg["by_tolerance"][t] = {
            "macro_f1": float(np.mean([r_["f1"] for r_ in rows])),
            "micro_precision": p,
            "micro_recall": r,
            "micro_f1": 2 * p * r / (p + r) if p + r else 0.0,
            "mean_abs_error": float(np.mean(errs)) if errs else None,
        }
    return agg
