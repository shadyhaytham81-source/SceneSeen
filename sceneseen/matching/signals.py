"""Cheap visual signals used next to the embedding: colour signature and crop quality."""
from __future__ import annotations

import numpy as np

COLOUR_KEY = "colour:hsv8x4x4:v1"
COLOUR_DIM = 128


def colour_signature(rgb: np.ndarray, suppress_background: bool = False) -> np.ndarray:
    """L1-normalised HSV histogram (8x4x4) of the central part of an image.

    suppress_background: for catalogue photos on a plain studio background, pixels close to the
    (uniform) border colour are left out so the signature describes the product, not the backdrop.
    """
    import cv2

    h, w = rgb.shape[:2]
    core = rgb[int(h * 0.15):max(int(h * 0.85), int(h * 0.15) + 1), int(w * 0.15):max(int(w * 0.85), int(w * 0.15) + 1)]
    core = np.ascontiguousarray(core)
    mask = None
    if suppress_background and h >= 16 and w >= 16:
        border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3), rgb[:, :3].reshape(-1, 3),
                                 rgb[:, -3:].reshape(-1, 3)]).astype(np.float32)
        if border.std(axis=0).mean() < 12:                      # plain backdrop
            dist = np.abs(core.astype(np.float32) - border.mean(axis=0)).sum(axis=2)
            m = (dist > 45).astype(np.uint8)
            if m.mean() > 0.05:                                 # keep the mask only if a product remains
                mask = m
    hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / max(float(hist.sum()), 1e-9)).astype(np.float32)


def colour_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Histogram intersection between every row of a [n, 128] and every row of b [m, 128] -> [n, m]."""
    out = np.zeros((len(a), len(b)), np.float32)
    for i in range(len(a)):                                     # n is tiny (occurrences of one object)
        out[i] = np.minimum(a[i][None, :], b).sum(axis=1)
    return out


def crop_weight(min_side: float, full: float = 160.0, floor: float = 0.25) -> float:
    """How much to trust one occurrence: small crops matched far worse in the calibration study
    (34 % top-1 under 160 px vs 59 % above), so they count less when several occurrences exist."""
    return float(min(1.0, max(floor, min_side / full)))
