"""Representative frames for commercial detection.

Phase 1 only keeps a 224-px thumbnail per shot, which is too small to find a watch or a phone.
Here one (or a few) frames per UNIQUE shot are extracted at detection resolution and cached as
JPEGs inside the video's cache folder. Frames are addressed by (shot id, relative position) and
identified by a content hash, so nothing downstream depends on machine-specific paths.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..media import VideoError, ffmpeg_exe, is_dataless
from ..shots import Shot

log = logging.getLogger(__name__)


def frame_time(shot: Shot, position: float, fps: float) -> float:
    """Timestamp inside the shot at a relative position (kept at least one frame from its edges)."""
    margin = min(1.0 / max(fps, 1.0), (shot.end - shot.start) / 4)
    return float(min(max(shot.start + position * (shot.end - shot.start), shot.start + margin), shot.end - margin))


def frame_name(shot_id: int, position: float) -> str:
    return f"shot_{shot_id:04d}_p{int(round(position * 100)):03d}.jpg"


def extract_frame(video: str | Path, t: float, long_side: int, out: Path) -> None:
    """One frame at time t, scaled so its longer side is `long_side` (never upscaled)."""
    vf = (f"scale='if(gte(iw,ih),min({long_side},iw),-2)':'if(gte(iw,ih),-2,min({long_side},ih))'"
          ":flags=lanczos")
    tmp = out.with_suffix(".part.jpg")
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-ss", f"{max(t, 0):.3f}", "-i", str(video),
           "-frames:v", "1", "-vf", vf, "-q:v", "3", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise VideoError(f"could not extract a frame at {t:.2f}s: {proc.stderr[-200:]}")
    tmp.replace(out)


def ensure_frames(video: str | Path, shots: list[Shot], shot_ids: list[int], positions: tuple[float, ...],
                  fps: float, long_side: int, frames_dir: Path, workers: int = 4) -> dict[tuple[int, float], Path]:
    """Extract missing frames (in parallel) and return {(shot_id, position): path} for all that exist."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    wanted = {(sid, p): frames_dir / frame_name(sid, p) for sid in shot_ids for p in positions}
    missing = [(k, path) for k, path in wanted.items() if not path.exists()]
    if missing:
        v = Path(video)
        if not v.exists() or is_dataless(v):
            raise VideoError("the video file is not available on this machine, so new frames cannot be extracted")

        def work(item):
            (sid, p), path = item
            try:
                extract_frame(v, frame_time(shots[sid], p, fps), long_side, path)
            except (VideoError, subprocess.TimeoutExpired) as e:
                log.warning("frame for shot %d@%.2f failed: %s", sid, p, e)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, missing))
    return {k: path for k, path in wanted.items() if path.exists()}


def image_hash(path: Path) -> str:
    """Content hash of a frame file (cache key component; independent of where the file lives)."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def load_rgb(path: Path) -> np.ndarray:
    """RGB uint8 array, or raises ValueError for unreadable / unsupported images."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"unreadable frame {Path(path).name}: {e}") from e
