"""Video ingestion: probing, stable video IDs and frame decoding.

All decoding goes through a bundled static ffmpeg binary (imageio-ffmpeg), so no
system-wide ffmpeg/ffprobe install is required.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


class VideoError(RuntimeError):
    """Raised for unreadable / unsupported input videos."""


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class VideoInfo:
    path: str
    video_id: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    video_codec: str
    size_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)


def video_id_for(path: str | Path) -> str:
    """Content-based ID (size + head + tail bytes) so renamed/re-uploaded files hit the cache."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    chunk = 4 * 1024 * 1024
    with p.open("rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, 0))
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VSTREAM = re.compile(r"Stream #\S+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+(?:\.\d+)?) (?:fps|tbr)")
_ROT = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def is_dataless(path: str | Path) -> bool:
    """True for macOS iCloud placeholders ("Optimize Mac Storage"): the bytes are not on disk
    and reading the file blocks until iCloud downloads it."""
    SF_DATALESS = 0x40000000
    try:
        return bool(getattr(Path(path).stat(), "st_flags", 0) & SF_DATALESS)
    except OSError:
        return False


def probe(path: str | Path) -> VideoInfo:
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"file not found: {p}")
    if is_dataless(p):
        raise VideoError(f"{p.name} is stored in iCloud only (not downloaded). Open it in Finder to download it.")
    proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(p)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, timeout=120)
    err = proc.stderr
    dm, vm = _DUR.search(err), _VSTREAM.search(err)
    if not vm:
        raise VideoError(f"no video stream found in {p.name} (is this a video file?)")
    duration = int(dm[1]) * 3600 + int(dm[2]) * 60 + float(dm[3]) if dm else 0.0
    vline = next(line for line in err.splitlines() if "Video:" in line)
    fm = _FPS.search(vline)
    fps = float(fm[1]) if fm else 0.0
    w, h = int(vm[2]), int(vm[3])
    rm = _ROT.search(err)
    if rm and abs(abs(float(rm[1])) - 90) < 1:  # ffmpeg auto-rotates on decode
        w, h = h, w
    if duration <= 0 or fps <= 0:
        raise VideoError(f"could not read duration/fps of {p.name}")
    return VideoInfo(
        path=str(p.resolve()),
        video_id=video_id_for(p),
        duration=round(duration, 3),
        fps=fps,
        width=w,
        height=h,
        has_audio=bool(re.search(r"Stream #\S+.*?: Audio:", err)),
        video_codec=vm[1],
        size_bytes=p.stat().st_size,
    )


def scaled_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Output (w, h) with the given short side, both even (yuv/rgb friendly)."""
    if width <= height:
        w, h = short_side, round(height * short_side / width)
    else:
        w, h = round(width * short_side / height), short_side
    return w + (w % 2), h + (h % 2)


def iter_frames(path: str | Path, width: int, height: int) -> Iterator[np.ndarray]:
    """Decode every frame (no frame dropping/duplication) as RGB uint8 [h, w, 3]."""
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
        "-an", "-sn", "-fps_mode", "passthrough",
        "-vf", f"scale={width}:{height}:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 8)
    finished = False
    n = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                finished = True
                break
            n += 1
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        if not finished:  # consumer stopped early: just stop ffmpeg
            proc.kill()
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        rc = proc.wait()
    if finished and rc != 0 and n == 0:
        raise VideoError(f"ffmpeg could not decode {Path(path).name}: {stderr[-500:]}")
    if finished and rc != 0:
        log.warning("ffmpeg reported errors after %d frames of %s: %s", n, Path(path).name, stderr[-300:])
