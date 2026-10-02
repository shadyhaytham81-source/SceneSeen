"""Ground-truth label files and the dev/test split.

ground_truth/<video_stem>.json:
    {
      "video": "video_01.mp4",
      "boundaries": [48.2, 127.4, 221.8],   # start time (s) of every scene except the first
      "annotator": "shady",
      "notes": "optional"
    }

ground_truth/splits.json:
    {"dev": ["video_01", ...], "test": ["video_05", ...]}

Labels must come from a human watching the video. Nothing in this module creates
boundaries on its own.
"""
from __future__ import annotations

import json
import logging
import random
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class LabelError(ValueError):
    pass


def validate_label(data: dict, duration: float | None = None) -> dict:
    if not isinstance(data, dict) or "video" not in data or "boundaries" not in data:
        raise LabelError("label must be an object with 'video' and 'boundaries'")
    b = data["boundaries"]
    if not isinstance(b, list) or not all(isinstance(x, (int, float)) for x in b):
        raise LabelError("'boundaries' must be a list of numbers (seconds)")
    if any(x <= 0 for x in b):
        raise LabelError("boundaries must be > 0 (the first scene implicitly starts at 0)")
    if sorted(b) != b or len(set(b)) != len(b):
        raise LabelError("boundaries must be strictly increasing")
    if duration is not None and b and b[-1] >= duration:
        raise LabelError(f"boundary {b[-1]} is beyond the video duration {duration:.2f}s")
    return data


def load_label(path: Path, duration: float | None = None) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LabelError(f"{path}: invalid JSON ({e})") from e
    try:
        return validate_label(data, duration)
    except LabelError as e:
        raise LabelError(f"{path}: {e}") from e


def save_label(gt_dir: Path, video_name: str, boundaries: list[float], annotator: str = "", notes: str = "") -> Path:
    gt_dir.mkdir(parents=True, exist_ok=True)
    data = validate_label({
        "video": video_name,
        "boundaries": [round(float(x), 3) for x in sorted(boundaries)],
        "annotator": annotator,
        "notes": notes,
    })
    path = gt_dir / f"{Path(video_name).stem}.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def nfc(s: str) -> str:
    """File names from macOS are often decomposed Unicode (NFD), while Windows/Linux and Git
    usually give composed names (NFC). Compare names only after normalising."""
    return unicodedata.normalize("NFC", s)


def find_video(videos_dir: Path, name: str) -> Path:
    """videos_dir/name, or a file whose name matches after Unicode normalisation."""
    p = videos_dir / name
    if p.exists() or not videos_dir.is_dir():
        return p
    target = nfc(name)
    for q in videos_dir.iterdir():
        if nfc(q.name) == target:
            return q
    return p


def _local_upload_copy(name: str, cache_dir: Path | None) -> Path | None:
    """An uploaded copy of the same file (recorded in cache/<id>/upload.json) that is on disk."""
    from .media import is_dataless

    if cache_dir is None:
        return None
    for up in cache_dir.glob("*/upload.json"):
        meta = json.loads(up.read_text(encoding="utf-8"))
        p = Path(meta.get("path", ""))
        if nfc(meta.get("name", "")) == nfc(name) and p.exists() and not is_dataless(p):
            return p
    return None


def labelled_videos(gt_dir: Path, videos_dir: Path, cache_dir: Path | None = None) -> list[tuple[str, Path, Path]]:
    """(stem, video_path, label_path) for every label whose video is readable. If the copy in
    videos_dir is an iCloud-only placeholder, a local uploaded copy of the same file is used."""
    from .media import is_dataless

    out = []
    for lp in sorted(gt_dir.glob("*.json")):
        if lp.name == "splits.json":
            continue
        name = json.loads(lp.read_text(encoding="utf-8")).get("video", "")
        vp = find_video(videos_dir, name)
        if vp.exists() and is_dataless(vp):
            alt = _local_upload_copy(name, cache_dir)
            if alt:
                log.warning("%s is iCloud-only; using local uploaded copy %s", name, alt.name)
                vp = alt
        if vp.suffix.lower() in VIDEO_EXTS and vp.exists() and not is_dataless(vp):
            out.append((nfc(lp.stem), vp, lp))
        else:
            log.warning("skipping label %s: video %s not found in %s", lp.name, name or "?", videos_dir)
    return out


def make_split(stems: list[str], test_fraction: float = 0.3, seed: int = 13) -> dict:
    """Deterministic random split. Test set gets at least 1 video when >= 2 exist."""
    stems = sorted(stems)
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(stems) * test_fraction)) if len(stems) >= 2 else 0
    return {"dev": sorted(shuffled[n_test:]), "test": sorted(shuffled[:n_test]), "seed": seed}


def load_or_create_split(gt_dir: Path, stems: list[str]) -> dict:
    """The split is frozen once written: new videos are appended to dev, never moved into test silently."""
    path = gt_dir / "splits.json"
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
        known = {nfc(s) for s in split.get("dev", []) + split.get("test", [])}
        new = [s for s in stems if nfc(s) not in known]
        if new:
            split["dev"] = sorted(split.get("dev", []) + new)
            path.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for k in ("dev", "test"):  # compare in NFC from here on (file itself is left as written)
            split[k] = [nfc(s) for s in split.get(k, [])]
        return split
    split = make_split(stems)
    path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split
