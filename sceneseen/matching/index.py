"""Vector index over catalogue reference images, with category gating.

`CatalogIndex` is an exact in-memory index (one matrix product per query batch). It is rebuilt
automatically when the catalogue changes. At 10,000 products x 3 images it searches in a few
milliseconds, so no approximate index is needed locally. The interface (build / search) is what
a PostgreSQL + pgvector implementation would provide in production:

    SELECT ... FROM image_embeddings JOIN products ...
    WHERE model_key = :key AND object_type = ANY(:compatible_types)
    ORDER BY embedding <=> :query LIMIT :shortlist
"""
from __future__ import annotations

import threading

import numpy as np
from sqlalchemy import func, select

from ..catalog.db import Brand, Database, ImageEmbedding, Product, ProductImage
from ..commercial import taxonomy as T
from .signals import COLOUR_DIM, COLOUR_KEY, colour_similarity
from .store import from_blob


class CatalogIndex:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.stamp = None
        self.E = np.zeros((0, 1), np.float32)       # [M, D] image embeddings (unit vectors)
        self.C = np.zeros((0, COLOUR_DIM), np.float32)
        self.product = np.zeros(0, np.int64)        # product id per row
        self.image = np.zeros(0, np.int64)          # image id per row
        self.types: dict[int, str | None] = {}      # product id -> object type
        self.categories: dict[int, str] = {}

    # ---- building
    @staticmethod
    def catalog_stamp(db: Database, model_key: str):
        """Cheap fingerprint of everything the index depends on."""
        with db.session() as s:
            return (s.scalar(select(func.count(ProductImage.id))),
                    s.scalar(select(func.count()).select_from(ImageEmbedding).where(ImageEmbedding.model_key == model_key)),
                    str(s.scalar(select(func.max(Product.updated_at)))),
                    s.scalar(select(func.count(Product.id)).where(Product.archived.is_(True))),
                    s.scalar(select(func.count(Brand.id)).where(Brand.archived.is_(True))))

    def build(self, db: Database) -> "CatalogIndex":
        stamp = self.catalog_stamp(db, self.model_key)
        with db.session() as s:
            rows = s.execute(
                select(ProductImage.id, ProductImage.product_id, ProductImage.sha256, Product.object_type,
                       Product.category, ImageEmbedding.vector)
                .join(Product, Product.id == ProductImage.product_id).join(Brand, Brand.id == Product.brand_id)
                .join(ImageEmbedding, (ImageEmbedding.sha256 == ProductImage.sha256)
                      & (ImageEmbedding.model_key == self.model_key))
                .where(Product.archived.is_(False), Brand.archived.is_(False))
                .order_by(ProductImage.product_id, ProductImage.id)).all()
            shas = {r[2] for r in rows}
            colour = {sha: from_blob(v) for sha, v in s.execute(
                select(ImageEmbedding.sha256, ImageEmbedding.vector).where(ImageEmbedding.model_key == COLOUR_KEY))
                if sha in shas} if shas else {}
        if rows:
            self.E = np.stack([from_blob(r[5]) for r in rows]).astype(np.float32)
            self.C = np.stack([colour.get(r[2], np.zeros(COLOUR_DIM, np.float32)) for r in rows])
        else:
            self.E, self.C = np.zeros((0, 1), np.float32), np.zeros((0, COLOUR_DIM), np.float32)
        self.image = np.array([r[0] for r in rows], np.int64)
        self.product = np.array([r[1] for r in rows], np.int64)
        self.types = {r[1]: r[3] for r in rows}
        self.categories = {r[1]: r[4] for r in rows}
        self._type_arr = np.array([r[3] or "" for r in rows])
        self._cat_arr = np.array([r[4] for r in rows])
        self.stamp = stamp
        return self

    def __len__(self) -> int:
        return len(self.product)

    @property
    def n_products(self) -> int:
        return len(self.types)

    # ---- gating
    def eligible(self, type_id: str) -> np.ndarray:
        """Rows a detection of `type_id` may be compared with: products of a compatible object
        type, or (for products saved without an object type) of the same commercial category."""
        if not len(self):
            return np.zeros(0, bool)
        obj = T.OBJECTS.get(type_id)
        if obj is None:
            return np.zeros(len(self), bool)
        compatible = list(T.compatible_types(type_id))
        return np.isin(self._type_arr, compatible) | ((self._type_arr == "") & (self._cat_arr == obj.category))

    # ---- search
    def search(self, Q: np.ndarray, Qc: np.ndarray, weights: np.ndarray, type_id: str, colour_weight: float,
               top_k: int = 5, shortlist: int = 200, gate: bool = True) -> dict:
        """Rank products for ONE detected object described by several occurrence crops.

        Q [n, D] crop embeddings, Qc [n, 128] crop colour signatures, weights [n] crop quality.
        score(crop, image)   = cosine + colour_weight * colour intersection
        score(crop, product) = best of the product's reference images
        score(object, product) = quality-weighted mean over the object's occurrence crops
        Returns {"eligible_products", "results": [{product_id, score, cosine, colour, image_id}]}.
        """
        mask = self.eligible(type_id) if gate else np.ones(len(self), bool)
        rows = np.where(mask)[0]
        n_eligible = len(set(self.product[rows].tolist()))
        if not len(rows) or not len(Q):
            return {"eligible_products": n_eligible, "results": []}
        cos = Q @ self.E[rows].T                                   # [n, m]
        if len(rows) > shortlist:                                  # two-stage: colour only for the best rows
            keep = np.unique(np.argpartition(-cos, shortlist - 1, axis=1)[:, :shortlist])
            rows, cos = rows[keep], cos[:, keep]
        col = colour_similarity(Qc, self.C[rows]) if colour_weight else np.zeros_like(cos)
        score = cos + colour_weight * col
        pid = self.product[rows]                                   # already sorted by product id
        starts = np.r_[0, np.where(np.diff(pid) != 0)[0] + 1]
        per_crop = np.maximum.reduceat(score, starts, axis=1)      # [n, products] best reference image
        w = np.asarray(weights, np.float64)
        w = w / w.sum()
        obj = (w[:, None] * per_crop).sum(axis=0)                  # [products]
        order = np.argsort(-obj)[:top_k]
        out = []
        for j in order:
            a, b = starts[j], (starts[j + 1] if j + 1 < len(starts) else len(pid))
            seg = score[:, a:b]
            flat = int(np.argmax((w[:, None] * seg).sum(axis=0)))
            out.append({"product_id": int(pid[a]), "score": float(obj[j]),
                        "cosine": float((w * cos[:, a + flat]).sum()), "colour": float((w * col[:, a + flat]).sum()),
                        "image_id": int(self.image[rows[a + flat]])})
        return {"eligible_products": n_eligible, "results": out}


_INDEXES: dict[tuple[str, str], CatalogIndex] = {}
_LOCK = threading.Lock()


def get_index(db: Database, model_key: str) -> CatalogIndex:
    """Shared index for (database, model), rebuilt only when the catalogue changed."""
    k = (db.url, model_key)
    with _LOCK:
        idx = _INDEXES.get(k)
        if idx is None or idx.stamp != CatalogIndex.catalog_stamp(db, model_key):
            idx = CatalogIndex(model_key).build(db)
            _INDEXES[k] = idx
        return idx
