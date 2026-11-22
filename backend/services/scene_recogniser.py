"""SceneRecogniser: suggest the show/episode from earlier tagged scans (FR4.1).

Nearest tagged frame embedding above a threshold. Works as-is on the in-memory store;
milestone 3 moves it into pgvector. Few suggestions at launch until tagged scans build up.
"""
from __future__ import annotations

import numpy as np

from ..schemas import Episode
from ..store import Store


class SceneRecogniser:
    def __init__(self, store: Store, threshold: float = 0.9):
        self.store, self.threshold = store, threshold

    def suggest(self, frame_vector: np.ndarray) -> Episode | None:
        best, best_sim = None, self.threshold
        for sid, vec in self.store.scan_vectors.items():
            scan = self.store.scans.get(sid)
            if not scan or not scan.episode_id or scan.flagged:
                continue
            sim = float(frame_vector @ vec)
            if sim >= best_sim:
                best, best_sim = scan.episode_id, sim
        return self.store.episodes.get(best) if best else None
