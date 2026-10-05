"""EXPERIMENTAL — DISABLED research (automatic product matching is not part of SceneSeen; docs/EXPERIMENTAL_MATCHING.md).

Catalogue scaling benchmark for Phase 2B (no model needed: synthetic embeddings in a real database).

    python scripts/matching_benchmark.py [--sizes 100 1000 10000] [--images 3]

For each catalogue size it creates a temporary SQLite catalogue with that many products (3 reference
images each, 512-d embeddings + colour signatures), then measures what a user actually waits for:
building the search index, ranking one detected object, ranking a whole video (40 objects), the
text search of the catalogue page, and memory. Writes reports/product_matching/scaling.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sceneseen.catalog import service as S                      # noqa: E402
from sceneseen.catalog.db import Brand, Database, ImageEmbedding, Product, ProductImage  # noqa: E402
from sceneseen.commercial import taxonomy as T                  # noqa: E402
from sceneseen.config import MatchingConfig                     # noqa: E402
from sceneseen.experimental.matching import matcher                          # noqa: E402
from sceneseen.experimental.matching.index import CatalogIndex, get_index    # noqa: E402
from sceneseen.experimental.matching.signals import COLOUR_DIM, COLOUR_KEY   # noqa: E402

KEY = "bench:512"


def build(db: Database, n: int, per: int, rng) -> None:
    types = [o for o in T.OBJECTS.values()]
    with db.session() as s:
        brands = [Brand(name=f"Brand {i}") for i in range(max(1, n // 50))]
        s.add_all(brands)
        s.flush()
        prods = [Product(brand_id=brands[i % len(brands)].id, name=f"Product {i:05d}", sku=f"SKU-{i:05d}",
                         category=types[i % len(types)].category, object_type=types[i % len(types)].id) for i in range(n)]
        s.add_all(prods)
        s.flush()
        E = rng.standard_normal((n * per, 512)).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        C = rng.random((n * per, COLOUR_DIM)).astype(np.float32)
        C /= C.sum(axis=1, keepdims=True)
        for i, p in enumerate(prods):
            for j in range(per):
                r = i * per + j
                sha = hashlib.sha256(f"{i}-{j}".encode()).hexdigest()
                s.add(ProductImage(product_id=p.id, role="front", sha256=sha, ext="jpg", width=800, height=800,
                                   size_bytes=1, position=j))
                s.add(ImageEmbedding(sha256=sha, model_key=KEY, dim=512, vector=E[r].tobytes()))
                s.add(ImageEmbedding(sha256=sha, model_key=COLOUR_KEY, dim=COLOUR_DIM, vector=C[r].tobytes()))


def timeit(fn, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    ap.add_argument("--images", type=int, default=3)
    a = ap.parse_args()
    cfg, rng, rows = MatchingConfig(), np.random.default_rng(0), []
    for n in a.sizes:
        with tempfile.TemporaryDirectory() as d:
            db = Database(f"sqlite:///{Path(d) / 'bench.db'}")
            t = time.perf_counter()
            build(db, n, a.images, rng)
            t_fill = time.perf_counter() - t
            tracemalloc.start()
            t_build, idx = timeit(lambda: CatalogIndex(KEY).build(db), repeat=1)
            mem = tracemalloc.get_traced_memory()[0] / 1e6
            tracemalloc.stop()
            t_stamp, _ = timeit(lambda: get_index(db, KEY))               # index reuse = one cheap fingerprint query
            # one object seen in 5 shots, of the most common type
            q = rng.standard_normal((5, 512)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            qc = rng.random((5, COLOUR_DIM)).astype(np.float32)
            qc /= qc.sum(axis=1, keepdims=True)
            sides = np.full(5, 240.0)
            t_obj, res = timeit(lambda: matcher.rank(idx, q, qc, sides, "sneakers", cfg), repeat=20)
            t_ungated, _ = timeit(lambda: idx.search(q, qc, np.ones(5), "sneakers", cfg.colour_weight, gate=False), repeat=10)
            types = list(T.OBJECTS)
            t_video, _ = timeit(lambda: [matcher.rank(idx, q, qc, sides, types[i % len(types)], cfg) for i in range(40)], repeat=5)
            t_text, found = timeit(lambda: S.search_products(db, q=f"{n // 2:05d}"[:4], limit=24))
            t_page, _ = timeit(lambda: S.search_products(db, category="fashion", limit=24, offset=48))
            rows.append({"products": n, "reference_images": n * a.images, "eligible_for_one_type": res["eligible_products"],
                         "index_build_s": round(t_build, 3), "index_memory_mb": round(mem, 1),
                         "index_reuse_check_ms": round(t_stamp * 1000, 2),
                         "rank_one_object_ms": round(t_obj * 1000, 2), "rank_one_object_ungated_ms": round(t_ungated * 1000, 2),
                         "rank_video_40_objects_ms": round(t_video * 1000, 1),
                         "catalog_text_search_ms": round(t_text * 1000, 2), "catalog_page_ms": round(t_page * 1000, 2),
                         "db_file_mb": round((Path(d) / "bench.db").stat().st_size / 1e6, 1), "fill_s": round(t_fill, 1)})
            print(json.dumps(rows[-1]))
            db.dispose()
    out = Path(__file__).resolve().parents[1] / "reports" / "product_matching" / "scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"embedding_dim": 512, "images_per_product": a.images, "object_types": len(T.OBJECTS),
                               "note": "synthetic embeddings; exact search (NumPy), category gating on", "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
