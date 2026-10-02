"""Shot -> scene grouping. Pure numpy, deterministic, cheap to re-run.

Method (windowed coherence; after Kender & Yeo 1998 and Hanjalic et al. 1999,
"logical story units", using modern CLIP embeddings as the shot descriptor):

1. Shot similarity S[i, j] = blend of CLIP cosine similarity (semantic: setting,
   people, time of day) and colour-histogram intersection (lighting/palette).
2. For every cut k (between shot k and k+1) we look at a window of shots on each
   side. Coherence(k) = mean of the top-k similarities between any left shot and
   any right shot (minus a small penalty for distance). Because the window spans
   several shots, A -> B -> A -> B dialogue patterns stay linked: the second A
   matches the first A across the cut, so the cut is not a scene boundary.
3. Boundary score = 1 - coherence. A cut becomes a scene boundary if its score is
   high in absolute terms AND it is a clear local peak: its TextTiling "depth"
   (Hearst 1997: how far the score rises above the valleys on either side) is large.
   Depth adapts to each film's editing style without assuming scene changes are
   rare. Cuts above a "strong" score bypass the depth test (so a run of several
   genuinely unrelated scenes in a row is not missed). A minimum scene duration is enforced by greedy non-maximum suppression.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .config import GroupingConfig
from .shots import Shot


# ---------------------------------------------------------------- similarity

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def clip_similarity(clip: np.ndarray) -> np.ndarray:
    """Cosine similarity of mean (re-normalised) frame embeddings per shot. clip: [N, F, D]."""
    emb = _normalize(_normalize(clip.astype(np.float32)).mean(axis=1))
    return emb @ emb.T


def color_similarity(color: np.ndarray) -> np.ndarray:
    """Histogram intersection of mean per-shot histograms. color: [N, F, B] (L1-normalised)."""
    h = color.astype(np.float32).mean(axis=1)
    h = h / np.maximum(h.sum(axis=1, keepdims=True), 1e-9)
    return np.minimum(h[:, None, :], h[None, :, :]).sum(axis=-1)


def similarity_matrix(clip: np.ndarray, color: np.ndarray, cfg: GroupingConfig) -> np.ndarray:
    cs = np.clip((clip_similarity(clip) - cfg.clip_floor) / (1.0 - cfg.clip_floor), 0.0, 1.0)
    hs = color_similarity(color)
    total = cfg.clip_weight + cfg.color_weight
    if total <= 0:
        raise ValueError("clip_weight + color_weight must be > 0")
    return (cfg.clip_weight * cs + cfg.color_weight * hs) / total


# ---------------------------------------------------------------- coherence

@dataclass
class CutScore:
    cut: int              # cut k is between shot k and shot k+1
    time: float           # start time of shot k+1
    coherence: float
    score: float          # 1 - coherence
    best_pair: tuple[int, int] | None
    depth: float = 0.0
    decision: str = ""    # boundary | below_abs | below_depth | min_duration

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best_pair"] = list(self.best_pair) if self.best_pair else None
        for k in ("time", "coherence", "score", "depth"):
            d[k] = round(float(d[k]), 4)
        return d


def cut_scores(S: np.ndarray, shots: list[Shot], cfg: GroupingConfig) -> list[CutScore]:
    n = len(shots)
    W = max(1, int(cfg.window_shots))
    out: list[CutScore] = []
    for k in range(n - 1):
        pairs: list[tuple[float, int, int]] = []
        left, right = range(max(0, k - W + 1), k + 1), range(k + 1, min(n, k + 1 + W))
        for i in left:
            for j in right:
                if shots[j].start - shots[i].end > cfg.window_seconds and not (i == k and j == k + 1):
                    continue  # too far apart in time to be linked (adjacent shots always compared)
                pairs.append((S[i, j] - cfg.distance_penalty * (j - i - 1), i, j))
        pairs.sort(reverse=True)
        # near the video edges one side has fewer shots, so fewer independent links are possible
        topk = max(1, min(int(cfg.coherence_topk), len(left), len(right)))
        top = pairs[:topk]
        coh = float(np.mean([p[0] for p in top]))
        out.append(CutScore(k, shots[k + 1].start, coh, 1.0 - coh, (top[0][1], top[0][2])))
    return out


def depth_scores(scores: np.ndarray) -> np.ndarray:
    """TextTiling depth: climb downhill from each point to the nearest valley on each
    side; depth = (peak - left valley) + (peak - right valley)."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    out = np.zeros(n)
    for k in range(n):
        lo = k
        while lo > 0 and s[lo - 1] <= s[lo]:
            lo -= 1
        hi = k
        while hi < n - 1 and s[hi + 1] <= s[hi]:
            hi += 1
        out[k] = (s[k] - s[lo]) + (s[k] - s[hi])
    return out


def window_depth_scores(scores: np.ndarray, radius: int) -> np.ndarray:
    """Prominence against the lowest cut score within `radius` cuts on each side:
    depth = (peak - min left) + (peak - min right), each side clipped at 0.

    Unlike TextTiling's downhill climb, a boundary whose immediate neighbour is ALSO a
    scene change (scenes of 1-3 shots, long takes) still gets credit for the ordinary
    within-scene cuts a little further away instead of a depth of ~0."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    out = np.zeros(n)
    for k in range(n):
        left = s[max(0, k - radius):k]
        right = s[k + 1:k + 1 + radius]
        out[k] = (max(0.0, s[k] - left.min()) if len(left) else 0.0) + \
                 (max(0.0, s[k] - right.min()) if len(right) else 0.0)
    return out


def select_boundaries(scores: list[CutScore], duration: float, cfg: GroupingConfig) -> list[CutScore]:
    """Mark decisions in-place and return the chosen boundaries (sorted by time)."""
    if not scores:
        return []
    arr = np.array([c.score for c in scores])
    if cfg.depth_mode == "window":
        depth = window_depth_scores(arr, max(1, cfg.window_shots // 2))
    elif cfg.depth_mode == "climb":
        depth = depth_scores(arr)
    else:
        raise ValueError(f"unknown depth_mode {cfg.depth_mode!r}")
    for c, d in zip(scores, depth):
        c.depth = float(d)
        c.decision = ("below_abs" if c.score < cfg.abs_threshold
                      else "" if c.score >= cfg.strong_threshold
                      else "below_depth" if d < cfg.depth_threshold else "")
    chosen: list[float] = []
    for c in sorted((c for c in scores if not c.decision), key=lambda c: -c.score):
        edges = [0.0, duration, *chosen]
        if min(abs(c.time - e) for e in edges) < cfg.min_scene_seconds:
            c.decision = "min_duration"
        else:
            c.decision = "boundary"
            chosen.append(c.time)
    return sorted((c for c in scores if c.decision == "boundary"), key=lambda c: c.time)


# ---------------------------------------------------------------- scenes

def boundaries_to_scenes(boundaries: list[float], duration: float, shots: list[Shot] | None = None) -> list[dict]:
    """Scene records from boundary times (each = start of a new scene)."""
    edges = [0.0, *sorted(b for b in boundaries if 0 < b < duration), duration]
    scenes = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        rec = {
            "scene_id": i + 1,
            "start_seconds": round(s, 3),
            "end_seconds": round(e, 3),
            "duration_seconds": round(e - s, 3),
        }
        if shots is not None:
            idx = [sh.index for sh in shots if sh.start >= s - 1e-3 and sh.start < e - 1e-3]
            rec["shot_start"] = idx[0] if idx else None
            rec["shot_end"] = idx[-1] if idx else None
            rec["shot_count"] = len(idx)
        scenes.append(rec)
    return scenes


def group_shots(
    shots: list[Shot], clip: np.ndarray, color: np.ndarray, duration: float, cfg: GroupingConfig
) -> dict:
    """Full grouping step. Returns boundaries, scenes and per-cut debug decisions."""
    if len(shots) <= 1:
        return {"boundaries": [], "scenes": boundaries_to_scenes([], duration, shots), "cuts": []}
    S = similarity_matrix(clip, color, cfg)
    scores = cut_scores(S, shots, cfg)
    chosen = select_boundaries(scores, duration, cfg)
    bounds = [round(c.time, 3) for c in chosen]
    return {
        "boundaries": bounds,
        "scenes": boundaries_to_scenes(bounds, duration, shots),
        "cuts": [c.to_dict() for c in scores],
        "adjacent_similarity": [round(float(S[k, k + 1]), 4) for k in range(len(shots) - 1)],
    }


# ---------------------------------------------------------------- baselines

def baseline_all_shots(shots: list[Shot]) -> list[float]:
    """Baseline A: every shot cut is a scene boundary."""
    return [s.start for s in shots[1:]]


def baseline_adjacent(shots, clip, color, duration, cfg: GroupingConfig) -> list[float]:
    """Baseline B: same similarity + selection, but only compares neighbouring shots
    (no temporal context). Isolates the value of the context window."""
    return group_shots(shots, clip, color, duration, cfg.replace(window_shots=1, coherence_topk=1))["boundaries"]
