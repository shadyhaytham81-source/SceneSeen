"""Detector: image -> garments and accessories (FR5).

STUB: returns nothing, so results come back empty. Milestone 3: wrap
sceneseen.commercial.detector (OWLv2, ~1 s per image on GPU) and keep detections whose
confidence passes `min_confidence` (target precision 85%, NFR6).
"""
from __future__ import annotations

from ..schemas import DetectedItem


class Detector:
    min_confidence = 0.25

    def detect(self, image: bytes) -> list[DetectedItem]:
        return []
