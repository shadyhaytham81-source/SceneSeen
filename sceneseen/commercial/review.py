"""Human review of commercial detections: Correct / Wrong / Missed.

Reviews are data, kept next to the scene labels in ground_truth/commercial_reviews/<video>.json
(committed to Git; no images). Each verdict stores a snapshot of what was judged (object type,
confidence, relevance, model, taxonomy version), so precision can be recomputed later for any
threshold without looking at the video again:

    precision of displayed detections = correct / (correct + wrong)

"Missed" entries record commercially relevant objects a reviewer saw that SceneSeen did not
show (a lower bound on what recall is losing; precision is the priority for now).
"""
from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

VERDICTS = ("correct", "wrong")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class ReviewStore:
    def __init__(self, gt_dir: Path, video_name: str):
        self.dir = gt_dir / "commercial_reviews"
        self.video = video_name
        self.path = self.dir / f"{_nfc(Path(video_name).stem)}.json"

    def load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                d.setdefault("reviews", {})
                d.setdefault("missed", [])
                return d
            except ValueError:
                pass
        return {"video": self.video, "reviews": {}, "missed": []}

    def _save(self, d: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def set_verdict(self, candidate: dict, verdict: str | None, reviewer: str, model: dict, taxonomy_version: str) -> dict:
        d = self.load()
        if verdict is None:
            d["reviews"].pop(candidate["key"], None)
        else:
            if verdict not in VERDICTS:
                raise ValueError(f"verdict must be one of {VERDICTS}")
            d["reviews"][candidate["key"]] = {
                "verdict": verdict, "type_id": candidate["type_id"], "label": candidate["label"],
                "category": candidate["category"], "scene_id": candidate["scene_id"],
                "detection_confidence": candidate["detection_confidence"],
                "commercial_relevance": candidate["commercial_relevance"], "displayed": candidate["displayed"],
                "shot_id": candidate["best"]["shot_id"], "box": candidate["best"]["box"],
                "frame_hash": candidate["best"]["frame_hash"], "model": model.get("name"),
                "detector_key": model.get("detector_key"), "taxonomy_version": taxonomy_version,
                "reviewer": reviewer, "reviewed": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._save(d)
        return d

    def add_missed(self, scene_id: int, label: str, reviewer: str, note: str = "") -> dict:
        d = self.load()
        d["missed"].append({"scene_id": scene_id, "label": label.strip(), "note": note, "reviewer": reviewer,
                            "reviewed": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._save(d)
        return d


def attach(result: dict, store_data: dict) -> dict:
    """Add each candidate's stored verdict (if any) and the scene's missed list to a result."""
    rv = store_data.get("reviews", {})
    for sc in result.get("scenes", []):
        for c in sc["candidates"]:
            c["review"] = rv.get(c["key"], {}).get("verdict")
        sc["missed"] = [m for m in store_data.get("missed", []) if m["scene_id"] == sc["scene_id"]]
    return result


def _prec(rows: list[dict]) -> dict:
    c = sum(r["verdict"] == "correct" for r in rows)
    w = sum(r["verdict"] == "wrong" for r in rows)
    return {"reviewed": c + w, "correct": c, "wrong": w, "precision": round(c / (c + w), 4) if c + w else None}


def summarize(gt_dir: Path, min_confidence: float | None = None, min_relevance: float = 0.45,
              floor: float = 0.40, detector_key: str | None = None) -> dict:
    """Precision of reviewed detections under the CURRENT display rules (per-type confidence tiers
    from the taxonomy, global floor, relevance threshold), overall and broken down. Reviews store
    each detection's confidence and relevance, so a different rule can be evaluated on the same
    reviews without watching anything again: pass `min_confidence` to try a flat threshold."""
    from .taxonomy import OBJECTS

    rows, missed, videos = [], 0, 0
    d = gt_dir / "commercial_reviews"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        videos += 1
        missed += len(data.get("missed", []))
        for r in data.get("reviews", {}).values():
            if detector_key and r.get("detector_key") != detector_key:
                continue
            rows.append({**r, "video": data.get("video", p.stem)})

    def shown(r, flat=None):
        tier = OBJECTS[r["type_id"]].min_confidence if r["type_id"] in OBJECTS else 1.0
        need = flat if flat is not None else max(floor, tier or 0.0)
        return r["detection_confidence"] >= need and r["commercial_relevance"] >= min_relevance

    disp = [r for r in rows if shown(r, min_confidence)]
    out = {"videos": videos, "missed_reported": missed, "all_reviewed": _prec(rows), "displayed": _prec(disp),
           "by_category": {}, "by_type": {}, "by_video": {}, "threshold_sweep": []}
    for key, field in (("by_category", "category"), ("by_type", "type_id"), ("by_video", "video")):
        for v in sorted({r[field] for r in disp}):
            out[key][v] = _prec([r for r in disp if r[field] == v])
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        sel = [r for r in rows if shown(r, t)]
        out["threshold_sweep"].append({"min_confidence": t, **_prec(sel)})
    return out
