"""Shot-to-shot visual similarity measures, vectorised.

All CLIP measures are cosine similarities of L2-normalised embeddings:

    cos(a, b) = (a · b) / (||a|| ||b||)      -> with unit vectors simply  a · b

so a whole similarity matrix is one matrix product (E @ E.T).

clip: [N, F, D] float array (N shots, F sampled frames per shot, D embedding dims)
"""
from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def clip_mean_similarity(clip: np.ndarray) -> np.ndarray:
    """Cosine similarity between the mean (re-normalised) frame embedding of each shot. [N, N]"""
    e = normalize(normalize(clip.astype(np.float32)).mean(axis=1))
    return e @ e.T


def clip_frame_similarity(clip: np.ndarray) -> np.ndarray:
    """All frame-to-frame cosine similarities. [N, N, F, F]: entry [i, j, a, b] compares frame a
    of shot i with frame b of shot j. One matrix product, no Python loops."""
    n, f, d = clip.shape
    e = normalize(clip.astype(np.float32)).reshape(n * f, d)
    return (e @ e.T).reshape(n, f, n, f).transpose(0, 2, 1, 3)


def clip_best_frame_similarity(frame_sim: np.ndarray) -> np.ndarray:
    """Best single frame pair (max). Tolerant, but one lucky frame is enough to match."""
    return frame_sim.max(axis=(2, 3))


def clip_matched_similarity(frame_sim: np.ndarray) -> np.ndarray:
    """Multi-frame correspondence (symmetric "chamfer" match): every sampled frame of shot i is
    matched to its most similar frame in shot j and vice versa; the result is the average.
    A shot only scores high if ALL of its sampled frames find a counterpart, which tolerates
    actor movement / small camera moves but rejects shots that merely share one similar moment."""
    return 0.5 * (frame_sim.max(axis=3).mean(axis=2) + frame_sim.max(axis=2).mean(axis=2))


def clip_worst_matched_similarity(frame_sim: np.ndarray) -> np.ndarray:
    """Strictest multi-frame measure: the worst of the best matches (min over frames)."""
    return np.minimum(frame_sim.max(axis=3).min(axis=2), frame_sim.max(axis=2).min(axis=2))


def color_similarity(color: np.ndarray) -> np.ndarray:
    """Histogram intersection of the mean HSV histogram per shot. color: [N, F, B]. -> [N, N]"""
    h = color.astype(np.float32).mean(axis=1)
    h = h / np.maximum(h.sum(axis=1, keepdims=True), 1e-9)
    return np.minimum(h[:, None, :], h[None, :, :]).sum(axis=-1)


# ---- pixel-level measures on one grayscale thumbnail per shot (used for the method comparison)

def phash_bits(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Perceptual hash (DCT of a 32x32 image, low-frequency block vs. its median). 64 bits."""
    import cv2

    small = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:hash_size, :hash_size]
    return (low > np.median(low)).ravel()


def hash_similarity(bits: np.ndarray) -> np.ndarray:
    """1 - normalised Hamming distance between hashes. bits: [N, K] bool. -> [N, N]"""
    b = bits.astype(np.float32)
    same = b @ b.T + (1 - b) @ (1 - b).T
    return same / bits.shape[1]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity of two equally sized grayscale images (Wang et al. 2004)."""
    import cv2

    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    va, vb, cov = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())
