"""Recommender (FR13).

STUB: most-clicked products, the popularity baseline the real model must beat (NFR8).
Milestone 3+: model developed on the H&M dataset (Recall@12, NDCG); users only ever see
partner products.
"""
from __future__ import annotations

from collections import Counter

from ..schemas import Product
from ..store import Store


class Recommender:
    def __init__(self, store: Store):
        self.store = store

    def recommend(self, user_id: str, k: int = 12) -> list[Product]:
        counts = Counter(c.product_id for c in self.store.clicks)
        counts.update(s.product_id for s in self.store.saved)
        ranked = [pid for pid, _ in counts.most_common()]
        ranked += [pid for pid in self.store.products if pid not in counts]
        return [self.store.products[pid] for pid in ranked[:k] if pid in self.store.products]
