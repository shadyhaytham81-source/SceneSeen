"""Per-shot visual features.

Each shot is represented by a few frames sampled at relative positions (default
15% / 50% / 85%), so camera moves and reveals inside a shot are covered without
decoding every frame. For every sampled frame we compute:

* an OpenCLIP image embedding (semantic content: place, setting, people, time of day)
* an HSV colour histogram (lighting / palette continuity)

The whole video is decoded once at low resolution and frames are embedded in
streaming batches, so memory stays flat for long videos.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from .config import FeaturesConfig
from .media import VideoInfo, iter_frames, scaled_size
from .shots import Shot

log = logging.getLogger(__name__)


def sample_frame_indices(shots: list[Shot], positions: tuple[float, ...]) -> np.ndarray:
    """[n_shots, n_positions] frame indices, each inside its shot."""
    out = np.zeros((len(shots), len(positions)), dtype=np.int64)
    for i, s in enumerate(shots):
        length = max(1, s.end_frame - s.start_frame)
        for j, p in enumerate(positions):
            out[i, j] = min(s.start_frame + int(p * length), s.end_frame - 1)
    return out


def color_histogram(frame_rgb: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / max(hist.sum(), 1e-9)).astype(np.float32)


def resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def load_clip(model_name: str, pretrained: str, device: str):
    import warnings

    import open_clip
    import torch

    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model, preprocess


class _ClipEmbedder:
    def __init__(self, cfg: FeaturesConfig):
        self.device = resolve_device(cfg.device)
        self.model, self.preprocess = load_clip(cfg.model, cfg.pretrained, self.device)

    def __call__(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(batch).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()


def extract_features(
    info: VideoInfo,
    shots: list[Shot],
    cfg: FeaturesConfig,
    thumbs_dir: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict[str, np.ndarray]:
    positions = tuple(cfg.frame_positions)
    idx = sample_frame_indices(shots, positions)
    wanted: dict[int, list[tuple[int, int]]] = {}
    for si in range(idx.shape[0]):
        for pj in range(idx.shape[1]):
            wanted.setdefault(int(idx[si, pj]), []).append((si, pj))
    mid = len(positions) // 2
    w, h = scaled_size(info.width, info.height, cfg.frame_short_side)

    embed = _ClipEmbedder(cfg)
    clip_out: np.ndarray | None = None
    color_out = np.zeros((len(shots), len(positions), int(np.prod(cfg.color_bins))), np.float32)
    pending_frames: list[np.ndarray] = []
    pending_slots: list[tuple[int, int]] = []

    def flush():
        nonlocal clip_out
        if not pending_frames:
            return
        emb = embed(pending_frames)
        if clip_out is None:
            clip_out = np.zeros((len(shots), len(positions), emb.shape[1]), np.float32)
        for (si, pj), e in zip(pending_slots, emb):
            clip_out[si, pj] = e
        pending_frames.clear()
        pending_slots.clear()

    if thumbs_dir:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
    last_needed = max(wanted) if wanted else -1
    total = max(1, last_needed + 1)
    for fi, frame in enumerate(iter_frames(info.path, w, h)):
        slots = wanted.get(fi)
        if slots:
            hist = color_histogram(frame, cfg.color_bins)
            for si, pj in slots:
                color_out[si, pj] = hist
                pending_frames.append(frame)
                pending_slots.append((si, pj))
                if thumbs_dir and pj == mid:
                    _save_thumb(frame, thumbs_dir / f"shot_{si:04d}.jpg")
            if len(pending_frames) >= cfg.batch_size:
                flush()
        if progress and fi % 250 == 0:
            progress(min(fi / total, 1.0))
        if fi >= last_needed:
            break
    flush()
    if clip_out is None:
        raise RuntimeError("no frames decoded for feature extraction")
    # Shots whose sample frames were never decoded (e.g. truncated file): copy neighbours.
    missing = np.where(np.abs(clip_out).sum(axis=(1, 2)) == 0)[0]
    for si in missing:
        src = si - 1 if si > 0 else si + 1
        if 0 <= src < len(shots):
            clip_out[si], color_out[si] = clip_out[src], color_out[src]
    if len(missing):
        log.warning("%d shots had no decodable sample frames; copied neighbours", len(missing))
    if progress:
        progress(1.0)
    return {"clip": clip_out.astype(np.float16), "color": color_out, "frame_indices": idx}


def _save_thumb(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path, quality=80)
