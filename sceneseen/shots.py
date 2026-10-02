"""Shot boundary detection.

A shot list is represented by its *cut frames*: the frame indices where a new shot
starts (excluding frame 0). Everything else (times, shot records) derives from that,
which keeps the deterministic post-processing small and testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np

from .config import ShotsConfig
from .media import VideoInfo, iter_frames

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Shot:
    index: int
    start_frame: int
    end_frame: int  # exclusive
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- pure helpers

def predictions_to_cuts(pred: np.ndarray, threshold: float) -> list[int]:
    """Turn per-frame transition probabilities into cut frames.

    A run of frames above threshold is one transition (one frame for a hard cut,
    several for a dissolve/fade). The new shot starts in the middle of the run
    (for a 1-frame run at i, that is frame i+1).
    """
    above = np.asarray(pred) > threshold
    cuts, i, n = [], 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            cut = (i + j + 2) // 2
            if 0 < cut < n:
                cuts.append(cut)
            i = j + 1
        else:
            i += 1
    return cuts


def merge_short_shots(cuts: list[int], n_frames: int, min_frames: int) -> list[int]:
    """Drop cuts that would create shots shorter than `min_frames`.

    Short shots are absorbed into the previous shot (or the next one at the start).
    """
    cuts = sorted(c for c in set(cuts) if 0 < c < n_frames)
    if min_frames <= 1:
        return cuts
    kept: list[int] = []
    prev = 0
    for c in cuts:
        if c - prev >= min_frames:
            kept.append(c)
            prev = c
    # last shot too short -> remove the last cut
    while kept and n_frames - kept[-1] < min_frames:
        kept.pop()
    return kept


def cuts_to_shots(cuts: list[int], n_frames: int, fps: float, duration: float) -> list[Shot]:
    edges = [0, *cuts, n_frames]
    shots = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        end_t = duration if i == len(edges) - 2 else e / fps
        shots.append(Shot(i, s, e, round(s / fps, 3), round(end_t, 3)))
    return shots


# ---------------------------------------------------------------- detectors

def _detect_transnet(info: VideoInfo, cfg: ShotsConfig) -> tuple[list[int], int, np.ndarray]:
    import torch
    from transnetv2_pytorch import TransNetV2

    frames = np.stack(list(iter_frames(info.path, 48, 27)))  # [N, 27, 48, 3]
    model = TransNetV2(device=cfg.transnet_device)
    with torch.no_grad():
        single, _ = model.predict_frames(torch.from_numpy(frames).to(model.device), quiet=True)
    pred = single.cpu().numpy().astype(np.float32)
    return predictions_to_cuts(pred, cfg.transnet_threshold), len(frames), pred


def _detect_pyscenedetect(info: VideoInfo, cfg: ShotsConfig) -> tuple[list[int], int, None]:
    from scenedetect import AdaptiveDetector, detect

    scenes = detect(info.path, AdaptiveDetector(adaptive_threshold=cfg.pyscenedetect_threshold), show_progress=False)
    if not scenes:
        n = round(info.duration * info.fps)
        return [], n, None
    n = scenes[-1][1].get_frames()
    return [s[0].get_frames() for s in scenes[1:]], n, None


def detect_shots(info: VideoInfo, cfg: ShotsConfig) -> dict:
    """Run the configured detector. Returns a JSON-serialisable record."""
    if cfg.detector == "transnetv2":
        cuts, n_frames, pred = _detect_transnet(info, cfg)
    elif cfg.detector == "pyscenedetect":
        cuts, n_frames, pred = _detect_pyscenedetect(info, cfg)
    else:
        raise ValueError(f"unknown shot detector: {cfg.detector}")
    raw_count = len(cuts) + 1
    cuts = merge_short_shots(cuts, n_frames, max(1, round(cfg.min_shot_seconds * info.fps)))
    shots = cuts_to_shots(cuts, n_frames, info.fps, info.duration)
    log.info("shots: %d detected (%d before short-shot merge), %d frames", len(shots), raw_count, n_frames)
    return {
        "detector": cfg.detector,
        "n_frames": n_frames,
        "raw_shot_count": raw_count,
        "shots": [s.to_dict() for s in shots],
        "transition_probs": None if pred is None else [round(float(p), 4) for p in pred],
    }


def shots_from_record(rec: dict) -> list[Shot]:
    return [Shot(**s) for s in rec["shots"]]
