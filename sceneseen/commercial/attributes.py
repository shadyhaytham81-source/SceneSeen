"""Simple visual attributes of a detected object. V1: dominant colour name ("white sneakers").

Deliberately conservative: a colour is only reported when one colour clearly dominates the
centre of the box; otherwise None. This describes appearance, it never identifies a product.
"""
from __future__ import annotations

import numpy as np


def dominant_color(rgb: np.ndarray, box: list[float], min_fraction: float = 0.55) -> str | None:
    import cv2

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    # central 60 % of the box: skips background at the edges
    mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    xa, xb = int((x1 + mx) * w), int((x2 - mx) * w)
    ya, yb = int((y1 + my) * h), int((y2 - my) * h)
    if xb - xa < 4 or yb - ya < 4:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[ya:yb, xa:xb]), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    names = np.full(len(hue), "", dtype=object)
    names[val < 0.18] = "black"
    grey = (names == "") & (sat < 0.16)
    names[grey & (val > 0.78)] = "white"
    names[grey & (val <= 0.78)] = "grey"
    col = names == ""
    for name, lo, hi in (("red", 0, 15), ("orange", 15, 40), ("yellow", 40, 70), ("green", 70, 165),
                         ("blue", 165, 260), ("purple", 260, 300), ("pink", 300, 340), ("red", 340, 361)):
        names[col & (hue >= lo) & (hue < hi)] = name
    warm = np.isin(names, ["orange", "yellow"])
    names[warm & (val < 0.55) & (hue < 50)] = "brown"
    names[warm & (sat < 0.35) & (val >= 0.55)] = "beige"
    vals, counts = np.unique(names, return_counts=True)
    k = int(np.argmax(counts))
    return str(vals[k]) if counts[k] / len(names) >= min_fraction else None
