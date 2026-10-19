"""Plain data records passed between the Phase 2A stages (all JSON-serialisable).

Boxes are [x1, y1, x2, y2] as FRACTIONS of the frame (0-1), so they do not depend on the
resolution a frame happened to be extracted at.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RawDetection:
    """What the detector returned for one query in one frame (before any SceneSeen logic)."""
    type_id: str            # taxonomy.OBJECTS key
    prompt: str             # the text query that fired (raw detector label)
    score: float            # detector confidence
    box: list[float]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 4)
        d["box"] = [round(float(v), 4) for v in self.box]
        return d


@dataclass
class Detection(RawDetection):
    """A detection placed in the video: which shot / frame it came from."""
    shot_id: int = -1
    position: float = 0.5
    frame_hash: str = ""
    color: str | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


@dataclass
class Candidate:
    """One commercial object in one scene, de-duplicated across repeated shots."""
    key: str
    scene_id: int
    type_id: str
    label: str
    category: str
    category_name: str
    icon: str
    detection_confidence: float
    commercial_relevance: float
    displayed: bool
    hidden_reason: str | None
    seen_count: int                      # shot occurrences (incl. repeats of the same camera set-up)
    detected_in_unique_shots: int        # unique shots whose representative frame was actually analysed
    first_seen: float
    last_seen: float
    screen_time: float
    max_instances: int                   # most instances visible in a single frame
    color: str | None
    best: dict                           # {shot_id, position, box, frame_hash}: thumbnail / crop source
    occurrences: list[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
