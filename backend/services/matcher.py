"""Matcher: detected item -> labelled products (FR6-FR8).

The labelling rules are final: same category only, Exact only when an approved
VerifiedItem exists for the scan's episode, nothing below the cutoff. The search itself is
a brute-force cosine over the in-memory store; milestone 3 moves it into a pgvector query.
"""
from __future__ import annotations

import numpy as np

from ..schemas import DetectedItem, MatchLabel, MatchResult, Product
from ..store import Store

NO_MATCH = "no close match yet"


class Matcher:
    def __init__(self, store: Store, cutoff: float = 0.25, k: int = 5):
        self.store, self.cutoff, self.k = store, cutoff, k

    def candidates(self, item: DetectedItem, vector: np.ndarray, brand_id: str | None = None) -> list[tuple[Product, float]]:
        out = []
        for pid, p in self.store.products.items():
            if p.category != item.category or (brand_id and p.brand_id != brand_id):
                continue
            out.append((p, float(vector @ self.store.product_vectors[pid])))
        return sorted(out, key=lambda t: t[1], reverse=True)

    def find_similar(self, item: DetectedItem, vector: np.ndarray, episode_id: str | None = None) -> list[MatchResult]:
        exact_ids = self.store.approved_products(episode_id)
        kept = [(p, s) for p, s in self.candidates(item, vector) if s >= self.cutoff or p.id in exact_ids]
        # confirmed products first, then by similarity
        kept.sort(key=lambda t: (t[0].id not in exact_ids, -t[1]))
        results = []
        for rank, (p, s) in enumerate(kept[: self.k], start=1):
            label = MatchLabel.exact if p.id in exact_ids else MatchLabel.similar
            m = MatchResult(rank=rank, similarity=round(s, 4), label=label, product=p, link="")
            m.link = f"/r/{m.id}"
            results.append(m)
        return results
