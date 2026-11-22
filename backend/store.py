"""In-memory store: a stand-in for Supabase (PostgreSQL + pgvector) until milestone 3.

Same shape as db/schema.sql, so swapping it for a Supabase client only changes this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import Brand, Click, Episode, MatchResult, Product, SavedItem, Scan, Show, VerifiedItem, ItemResults


@dataclass
class Store:
    shows: dict[str, Show] = field(default_factory=dict)
    episodes: dict[str, Episode] = field(default_factory=dict)
    brands: dict[str, Brand] = field(default_factory=dict)
    products: dict[str, Product] = field(default_factory=dict)
    product_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    scans: dict[str, Scan] = field(default_factory=dict)
    scan_vectors: dict[str, np.ndarray] = field(default_factory=dict)     # frame embeddings (scene recognition)
    scan_results: dict[str, list[ItemResults]] = field(default_factory=dict)
    matches: dict[str, MatchResult] = field(default_factory=dict)
    verified: dict[str, VerifiedItem] = field(default_factory=dict)
    clicks: list[Click] = field(default_factory=list)
    saved: list[SavedItem] = field(default_factory=list)
    views: list[tuple[str, str]] = field(default_factory=list)            # (product_id, scan_id)

    def approved_products(self, episode_id: str | None) -> set[str]:
        if not episode_id:
            return set()
        return {v.product_id for v in self.verified.values()
                if v.episode_id == episode_id and v.status == "approved"}


store = Store()
