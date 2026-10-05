"""Embedding caches.

* Catalogue images: table `image_embeddings`, one row per (image content hash, model key).
  An unchanged image is never embedded twice with the same model; a new model or model
  revision has a different key and gets its own rows.
* Detected-object crops: one small .npz per video and model key inside the video's commercial
  cache, keyed by (frame content hash, box). Re-running matching re-uses them.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from sqlalchemy import select

from ...catalog.db import Database, ImageEmbedding, Product, ProductImage
from ...catalog.images import ImageError, ImageStore
from .signals import COLOUR_KEY, colour_signature

log = logging.getLogger(__name__)


def _to_blob(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def missing_catalog_images(db: Database, model_key: str) -> list[tuple[str, str]]:
    """(sha256, ext) of active catalogue images without an embedding for this key."""
    with db.session() as s:
        have = select(ImageEmbedding.sha256).where(ImageEmbedding.model_key == model_key)
        rows = s.execute(select(ProductImage.sha256, ProductImage.ext).join(Product)
                         .where(Product.archived.is_(False), ProductImage.sha256.not_in(have)).distinct()).all()
        return [(r[0], r[1]) for r in rows]


def ensure_catalog_embeddings(db: Database, store: ImageStore, embedder, progress: Callable[[float], None] | None = None,
                              batch: int = 32) -> dict:
    """Embed catalogue images that have no embedding yet (model embedding + colour signature).
    Unreadable or missing image files are skipped and reported, never fatal."""
    embedder.load()
    todo = missing_catalog_images(db, embedder.key)
    todo_colour = missing_catalog_images(db, COLOUR_KEY)
    done, bad = 0, []
    for sha, ext in todo_colour:
        try:
            sig = colour_signature(store.load_rgb(sha, ext), suppress_background=True)
        except ImageError as e:
            bad.append({"sha256": sha, "error": str(e)})
            continue
        with db.session() as s:
            s.merge(ImageEmbedding(sha256=sha, model_key=COLOUR_KEY, dim=len(sig), vector=_to_blob(sig)))
    for i in range(0, len(todo), batch):
        chunk, images = [], []
        for sha, ext in todo[i:i + batch]:
            try:
                images.append(store.load_rgb(sha, ext))
                chunk.append(sha)
            except ImageError as e:
                bad.append({"sha256": sha, "error": str(e)})
        if images:
            vecs = embedder.embed(images)
            with db.session() as s:
                for sha, v in zip(chunk, vecs):
                    s.merge(ImageEmbedding(sha256=sha, model_key=embedder.key, dim=len(v), vector=_to_blob(v)))
            done += len(chunk)
        if progress:
            progress(min(1.0, (i + batch) / max(1, len(todo))))
    bad_unique = list({b["sha256"]: b for b in bad}.values())
    if bad_unique:
        log.warning("%d catalogue image(s) could not be embedded", len(bad_unique))
    return {"embedded": done, "already_cached": None, "failed": bad_unique, "model_key": embedder.key}


class CropCache:
    """Embeddings + colour signatures of detected-object crops for one video and one model key."""

    def __init__(self, commercial_dir: Path, model_key: str):
        tag = hashlib.sha1(model_key.encode()).hexdigest()[:10]
        self.path = commercial_dir / f"crop_embeddings_{tag}.npz"
        self.keys: list[str] = []
        self.emb = self.col = self.side = None
        if self.path.exists():
            try:
                d = np.load(self.path, allow_pickle=False)
                if str(d["model_key"]) == model_key:
                    self.keys, self.emb, self.col, self.side = list(d["keys"]), d["emb"], d["colour"], d["side"]
            except Exception as e:   # corrupted cache: start again
                log.warning("ignoring unreadable crop cache %s (%s)", self.path.name, e)
        self.model_key = model_key
        self._index = {k: i for i, k in enumerate(self.keys)}

    @staticmethod
    def key(frame_hash: str, box: list[float]) -> str:
        return f"{frame_hash}|" + ",".join(f"{v:.4f}" for v in box)

    def get(self, key: str):
        i = self._index.get(key)
        return None if i is None else (self.emb[i], self.col[i], float(self.side[i]))

    def add(self, keys: list[str], emb: np.ndarray, col: np.ndarray, side: np.ndarray) -> None:
        if not keys:
            return
        self.emb = emb if self.emb is None else np.concatenate([self.emb, emb])
        self.col = col if self.col is None else np.concatenate([self.col, col])
        self.side = side if self.side is None else np.concatenate([self.side, side])
        for k in keys:
            self._index[k] = len(self.keys)
            self.keys.append(k)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, model_key=np.array(self.model_key), keys=np.array(self.keys), emb=self.emb,
                            colour=self.col, side=self.side)
        tmp.replace(self.path)
