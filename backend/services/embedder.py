"""Embedder: image -> unit vector.

STUB: a deterministic vector from the image bytes, so equal images match and tests run
without downloading a model. Milestone 3: wrap sceneseen.commercial.matching.ClipEncoder
(OpenCLIP), the same encoder already used for the catalogue index.
"""
from __future__ import annotations

import hashlib

import numpy as np

DIM = 512


class Embedder:
    dim = DIM

    def embed(self, image: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(image).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)
