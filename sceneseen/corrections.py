"""Manual scene corrections: split a scene, merge adjacent scenes.

Corrections are stored per video as the full corrected boundary list plus an
append-only log of operations (useful as future training signal: every merge is
a "false boundary" example, every split a "missed boundary" example).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class CorrectionError(ValueError):
    pass


def merge(boundaries: list[float], scene_index: int) -> list[float]:
    """Merge scene `scene_index` (0-based) with the next scene."""
    b = sorted(boundaries)
    if not 0 <= scene_index < len(b):
        raise CorrectionError(f"cannot merge scene {scene_index + 1} with a following scene")
    return b[:scene_index] + b[scene_index + 1:]


def split(boundaries: list[float], at: float, duration: float, cut_times: list[float] | None = None,
          snap_before: float = 2.0, snap_after: float = 0.5) -> tuple[list[float], float]:
    """Insert a boundary at `at`, snapped to the nearest shot cut in [at - snap_before, at + snap_after].

    The window is asymmetric because people pause *after* seeing the new scene start: in the
    first 6 real labels, every off-cut label was 1.0-1.6 s later than the actual cut."""
    if cut_times:
        near = [c for c in cut_times if at - snap_before <= c <= at + snap_after]
        if near:
            at = min(near, key=lambda c: abs(c - at))
    at = round(float(at), 3)
    if not 0 < at < duration:
        raise CorrectionError("split point must be inside the video")
    if any(abs(at - x) < 0.25 for x in boundaries):
        raise CorrectionError("a scene boundary already exists there")
    return sorted([*boundaries, at]), at


class CorrectionStore:
    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "corrections.json"

    def load(self) -> dict | None:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def save(self, boundaries: list[float], op: dict, predicted: list[float]) -> dict:
        data = self.load() or {"predicted_boundaries": predicted, "log": []}
        data["boundaries"] = sorted(round(float(x), 3) for x in boundaries)
        data["log"].append({**op, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return data

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
