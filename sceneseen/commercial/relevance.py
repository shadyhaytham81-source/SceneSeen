"""Commercial relevance, V1: deterministic rules (no model).

relevance = base value of the object type  x  how well this scene shows it

    base        taxonomy value: would a brand pay for / a viewer shop for this kind of thing?
    size        large enough to be recognisable, not so large that it is just "the background"
    centrality  nearer the centre of the frame = more likely the subject
    persistence seen in several shot occurrences of the scene, not a single glimpse

The presentation factor only scales the base value between 60 % and 100 %, so a prominent chair
never outranks a visible watch, but a tiny background chair drops below the display threshold.
"""
from __future__ import annotations

import math

from .taxonomy import ObjectType

W_FLOOR, W_SIZE, W_CENTER, W_PERSIST = 0.60, 0.20, 0.08, 0.12


def size_score(area: float) -> float:
    """0-1. Full score from 1 % of the frame; fades above 60 % (object fills the frame)."""
    if area <= 0:
        return 0.0
    s = min(1.0, math.sqrt(area / 0.01))
    if area > 0.6:
        s *= max(0.3, 1.0 - (area - 0.6) / 0.4 * 0.7)
    return s


def center_score(box: list[float]) -> float:
    cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
    return max(0.0, 1.0 - math.hypot(cx, cy) / 0.7071)


def persistence_score(seen_count: int) -> float:
    return min(1.0, seen_count / 3.0)


def commercial_relevance(obj: ObjectType, box: list[float], seen_count: int) -> tuple[float, dict]:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    f = {"base": obj.base_relevance, "size": round(size_score(area), 3), "centrality": round(center_score(box), 3),
         "persistence": round(persistence_score(seen_count), 3)}
    presentation = W_FLOOR + W_SIZE * f["size"] + W_CENTER * f["centrality"] + W_PERSIST * f["persistence"]
    return round(obj.base_relevance * presentation, 4), f
