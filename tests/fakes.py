"""Shared test doubles for Phase 2B (no model download, no network)."""
import io

import numpy as np
from PIL import Image

from sceneseen.experimental.matching.embedder import EmbedderUnavailable


class FakeEmbedder:
    """Deterministic "embedding": the image shrunk to 6x6 RGB, centred and normalised.
    Identical pictures get identical vectors; different pictures get different ones."""

    def __init__(self, key="fake:test:v1", fail=False):
        self.key, self.fail = key, fail
        self.device, self.dim, self.calls, self.images = None, 108, 0, 0

    def cache_key(self):
        return self.key

    def load(self):
        if self.fail:
            raise EmbedderUnavailable("could not load fake model: ConnectionError: offline")
        self.device = "cpu"

    def embed(self, images):
        self.load()
        self.calls += 1
        self.images += len(images)
        out = []
        for im in images:
            v = np.asarray(Image.fromarray(im).resize((6, 6), Image.BILINEAR), np.float32).ravel() / 255.0
            v = v - v.mean()
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out).astype(np.float32)

    def describe(self):
        return {"key": self.key, "name": "fake", "model": "fake", "license": "test", "device": self.device, "dim": self.dim,
                "load_seconds": 0.0}


def pattern(seed: int, size: int = 240) -> np.ndarray:
    """A distinctive 3x3 colour-block picture per seed (a stand-in for a product photo)."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (3, 3, 3), dtype=np.uint8)
    return np.kron(blocks, np.ones((size // 3, size // 3, 1), np.uint8))


def jpeg(arr: np.ndarray, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, quality=95, subsampling=0) if fmt == "JPEG" else Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()
