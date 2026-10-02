"""Scene exports: JSON metadata and per-scene video clips.

Clip cutting: "reencode" (default) is frame-accurate. It was verified against
the source: exact frame count, first frame = the scene's first source frame.
"copy" is a lossless, near-instant stream copy, but it can only cut on
keyframes, and in testing it was off by a few frames even when the scene started
on a keyframe (B-frame reordering, edit lists). Use it only when speed matters
more than exact scene edges.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Callable

from .config import ExportConfig
from .media import ffmpeg_exe

log = logging.getLogger(__name__)


def write_scene_json(result_public: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_public, indent=2) + "\n", encoding="utf-8")
    return path


def _clip_cmd(src: str, dst: Path, start: float, dur: float, copy: bool, cfg: ExportConfig) -> list[str]:
    base = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y"]
    if copy:
        return base + ["-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a?",
                       "-c", "copy", str(dst)]
    return base + ["-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "0:a?",
                   "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)]


def export_clips(
    src: str | Path,
    scenes: list[dict],
    out_dir: Path,
    fps: float,
    cfg: ExportConfig,
    progress: Callable[[float], None] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if cfg.clip_mode not in ("reencode", "copy"):
        raise ValueError(f"unknown clip_mode {cfg.clip_mode!r} (use 'reencode' or 'copy')")
    files, methods = [], []
    for n, s in enumerate(scenes):
        start, dur = s["start_seconds"], s["duration_seconds"]
        copy = cfg.clip_mode == "copy"
        dst = out_dir / f"scene_{s['scene_id']:03d}.mp4"
        proc = subprocess.run(_clip_cmd(str(src), dst, start, dur, copy, cfg), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0 and copy:
            log.warning("stream copy failed for scene %s, re-encoding: %s", s["scene_id"], proc.stderr[-200:])
            copy = False
            proc = subprocess.run(_clip_cmd(str(src), dst, start, dur, False, cfg), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on scene {s['scene_id']}: {proc.stderr[-400:]}")
        files.append(dst)
        methods.append("copy" if copy else "reencode")
        if progress:
            progress((n + 1) / len(scenes))
    summary = {
        "files": [str(f) for f in files],
        "copied": methods.count("copy"),
        "reencoded": methods.count("reencode"),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    log.info("exported %d clips (%d copied, %d re-encoded) in %.1fs", len(files), summary["copied"],
             summary["reencoded"], summary["seconds"])
    return summary


def zip_files(files: list[Path], extra: dict[str, str], zip_path: Path) -> Path:
    """Store-only zip (video is already compressed)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, Path(f).name)
        for name, text in extra.items():
            z.writestr(name, text)
    return zip_path
