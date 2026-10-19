"""De-duplication: from raw per-frame detections to one commercial candidate per object per scene.

1. inside a frame   overlapping boxes for the same thing are merged (non-maximum suppression):
                    same type, or competing types of one group (sneakers vs shoes), or any two
                    labels on virtually the same box.
2. inside a shot    detections from several sampled frames of the same shot are merged.
3. across a scene   Unique Shots tell us which shots are repeats of one camera set-up. The
                    representative's detections are credited to every occurrence of that set-up,
                    and all detections of one object type in a scene become ONE candidate with
                    its occurrence list ("Smartphone, seen in 6 shot occurrences, 00:31-00:58").
"""
from __future__ import annotations

from .models import Detection, iou
from .taxonomy import OBJECTS


def frame_nms(dets: list[Detection], same_iou: float = 0.5, group_iou: float = 0.6,
              any_iou: float = 0.85) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        gd = OBJECTS[d.type_id].group
        clash = False
        for k in kept:
            o = iou(d.box, k.box)
            if (k.type_id == d.type_id and o > same_iou) or (gd and gd == OBJECTS[k.type_id].group and o > group_iou) \
                    or o > any_iou:
                clash = True
                break
        if not clash:
            kept.append(d)
    return kept


def merge_shot_frames(per_frame: list[list[Detection]]) -> list[Detection]:
    """Union of the detections of several frames of one shot; the same object seen in two
    frames (same type, overlapping boxes) is kept once, with its best score."""
    return frame_nms([d for frame in per_frame for d in frame], same_iou=0.3, group_iou=0.5)


def max_instances(dets: list[Detection]) -> int:
    """Most instances of the object visible in any single frame."""
    per: dict[tuple[int, float], int] = {}
    for d in dets:
        per[(d.shot_id, d.position)] = per.get((d.shot_id, d.position), 0) + 1
    return max(per.values(), default=0)


def scene_groups(scene_unique: list[dict], dets_by_shot: dict[int, list[Detection]]) -> dict[str, dict]:
    """{type_id: {"detections": [...], "occurrences": [...]}} for one scene.

    scene_unique: the scene's unique-shot groups (from Phase 1 Unique Shots).
    dets_by_shot: detections of the representative shots that were analysed.
    """
    out: dict[str, dict] = {}
    for g in scene_unique:
        rep = g["representative_shot_id"]
        for d in dets_by_shot.get(rep, []):
            slot = out.setdefault(d.type_id, {"detections": [], "occurrences": {}, "unique_shots": set()})
            slot["detections"].append(d)
            slot["unique_shots"].add(g["unique_shot_id"])
            for occ in g["occurrences"]:
                # credited to every repeat of the camera set-up; "detected" marks the analysed frame
                slot["occurrences"].setdefault(occ["shot_id"], {
                    "shot_id": occ["shot_id"], "start": occ["start"], "end": occ["end"],
                    "unique_shot_id": g["unique_shot_id"], "detected": occ["shot_id"] == rep})
    for slot in out.values():
        slot["occurrences"] = sorted(slot["occurrences"].values(), key=lambda o: o["start"])
        slot["unique_shots"] = sorted(slot["unique_shots"])
    return out
