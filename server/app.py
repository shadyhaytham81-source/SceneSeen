"""SceneSeen web API + static UI.

    python -m sceneseen serve        ->  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sceneseen import grouping
from sceneseen.config import load_config
from sceneseen.corrections import CorrectionError, CorrectionStore, merge, split
from sceneseen.export import export_clips, zip_files
from sceneseen.ground_truth import VIDEO_EXTS, provenance_from_corrections, save_label
from sceneseen.media import VideoError, VideoInfo, ffmpeg_exe, probe, video_id_for
from sceneseen.pipeline import (STAGES, VideoCache, analyze, build_result, load_stage_outputs, public_result,
                                unique_shots_for)

from .jobs import Job, JobRunner

log = logging.getLogger("sceneseen.server")
CFG = load_config(*filter(None, os.environ.get("SCENESEEN_CONFIG", "").split(os.pathsep)))
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="SceneSeen", version="0.1.0")
jobs = JobRunner()

# overall progress = weighted stages
STAGE_SPAN = {"reading": (0.0, 0.02), "shots": (0.02, 0.6), "scenes": (0.6, 0.95), "results": (0.95, 1.0)}
BROWSER_CODECS = {"h264", "vp8", "vp9", "av1", "hevc"}
BROWSER_EXTS = {".mp4", ".m4v", ".mov", ".webm"}


# ---------------------------------------------------------------- helpers

def _cache(vid: str) -> VideoCache:
    if not vid.isalnum():
        raise HTTPException(400, "bad video id")
    return VideoCache(CFG.paths.cache_dir, vid)


def _meta(vid: str) -> dict:
    meta = _cache(vid).read_json("upload.json")
    if not meta:
        raise HTTPException(404, "unknown video")
    return meta


def _info(vid: str) -> VideoInfo:
    return VideoInfo(**_cache(vid).read_json("info.json"))


def _stage(vid: str) -> dict:
    """Cached shots/features. Uses the stored video info, so it also works when the uploaded
    file itself is temporarily unavailable (nothing is decoded when the cache is complete)."""
    _meta(vid)
    return load_stage_outputs(CFG, _info(vid))


def _current_result(vid: str, gcfg=None) -> dict:
    """Latest analysis with manual corrections applied (if any)."""
    cache = _cache(vid)
    result = cache.read_json("result.json")
    if result is None:
        raise HTTPException(404, "video has not been analysed yet")
    meta = _meta(vid)
    result["video"] = meta["name"]
    corr = CorrectionStore(cache.dir).load()
    if corr is not None and gcfg is None:
        from sceneseen.shots import Shot

        shots = [Shot(**s) for s in result["debug"]["shots"]]
        result["boundaries"] = corr["boundaries"]
        result["scenes"] = grouping.boundaries_to_scenes(corr["boundaries"], result["duration"], shots)
        result["scene_count"] = len(result["scenes"])
        result["corrected"] = True
        result["predicted_boundaries"] = corr["predicted_boundaries"]
    feat_key = result["debug"]["feature_key"]
    for s in result["scenes"]:
        if s.get("shot_start") is not None:
            mid = (s["shot_start"] + s["shot_end"]) // 2
            s["thumb"] = f"/api/videos/{vid}/thumbs/{feat_key}/{mid}"
    result["video_url"] = f"/api/videos/{vid}/media"
    result["has_ground_truth"] = (CFG.paths.ground_truth_dir / f"{Path(meta['name']).stem}.json").exists()
    return result


def _ensure_preview(info: VideoInfo, cache: VideoCache) -> Path:
    """Browsers cannot play e.g. MKV/AVI or MPEG-2; make a small H.264 proxy only when needed."""
    src = Path(info.path)
    if src.suffix.lower() in BROWSER_EXTS and info.video_codec in BROWSER_CODECS:
        return src
    proxy = cache.path("preview.mp4")
    if not proxy.exists():
        tmp = cache.path("preview.tmp.mp4")
        subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-vf", "scale=-2:480",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-c:a", "aac",
                        "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(proxy)
    return proxy


# ---------------------------------------------------------------- routes: upload & analysis

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext or '?'}'. Use one of: {', '.join(sorted(VIDEO_EXTS))}")
    CFG.paths.uploads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CFG.paths.uploads_dir, suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp, length=4 * 1024 * 1024)
        tmp_path = Path(tmp.name)
    vid = video_id_for(tmp_path)
    dest = CFG.paths.uploads_dir / f"{vid}{ext}"
    if dest.exists():
        tmp_path.unlink()
    else:
        tmp_path.replace(dest)
    try:
        info = probe(dest)
    except VideoError:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"'{Path(file.filename).name}' could not be read as a video. Is the file complete and playable?")
    cache = _cache(vid)
    cache.write_json("upload.json", {"name": Path(file.filename).name, "path": str(dest)})
    analysed = cache.read_json("result.json") is not None
    return {"video_id": vid, "name": Path(file.filename).name, "duration": info.duration, "analysed": analysed}


@app.post("/api/videos/{vid}/analyze")
def start_analysis(vid: str):
    meta = _meta(vid)
    cache = _cache(vid)

    def work(job: Job) -> dict:
        def progress(stage, frac):
            a, b = STAGE_SPAN[stage]
            job.stage, job.stage_label, job.progress = stage, STAGES[stage], a + (b - a) * frac

        result = analyze(meta["path"], CFG, progress)
        progress("results", 0.5)
        _ensure_preview(probe(meta["path"]), cache)
        return {"scene_count": result["scene_count"]}

    return jobs.submit(f"analyze-{vid}", "analyze", work).public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job.public()


@app.get("/api/videos")
def list_videos():
    out = []
    root = CFG.paths.cache_dir
    for d in sorted(root.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        up = json.loads((d.parent / "upload.json").read_text(encoding="utf-8")) if (d.parent / "upload.json").exists() else None
        if not up:
            continue
        r = json.loads(d.read_text(encoding="utf-8"))
        out.append({"video_id": d.parent.name, "name": up["name"], "duration": r["duration"],
                    "scene_count": r["scene_count"]})
    return out


@app.get("/api/videos/{vid}/result")
def get_result(vid: str, debug: bool = False):
    r = _current_result(vid)
    if not debug:
        r.pop("debug", None)
    else:
        r["debug"]["thumb_base"] = f"/api/videos/{vid}/thumbs/{r['debug']['feature_key']}/"
    return r


class RegroupBody(BaseModel):
    params: dict


@app.post("/api/videos/{vid}/regroup")
def regroup(vid: str, body: RegroupBody):
    """Developer mode: re-run only the grouping step with new parameters (cached features)."""
    try:
        gcfg = CFG.grouping.replace(**body.params)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    stage = _stage(vid)
    info = _info(vid)
    r = build_result(info, stage, gcfg, dict(stage["timings"]))
    r["video"] = _meta(vid)["name"]
    feat_key = r["debug"]["feature_key"]
    for s in r["scenes"]:
        if s.get("shot_start") is not None:
            s["thumb"] = f"/api/videos/{vid}/thumbs/{feat_key}/{(s['shot_start'] + s['shot_end']) // 2}"
    r["debug"]["thumb_base"] = f"/api/videos/{vid}/thumbs/{feat_key}/"
    r["video_url"] = f"/api/videos/{vid}/media"
    r["preview_only"] = True
    return r


@app.get("/api/videos/{vid}/unique-shots")
def unique_shots(vid: str):
    """Repeated-shot groups for the current scenes (post-processing; never changes scenes)."""
    result = _current_result(vid)
    rec = unique_shots_for(CFG, _info(vid), _stage(vid), result["scenes"])
    rec["thumb_base"] = f"/api/videos/{vid}/thumbs/{result['debug']['feature_key']}/"
    return rec


# ---------------------------------------------------------------- routes: media

@app.get("/api/videos/{vid}/media")
def media(vid: str):
    info = _info(vid)
    path = _ensure_preview(info, _cache(vid))
    return FileResponse(path, media_type="video/mp4" if path.suffix != ".webm" else "video/webm")


@app.get("/api/videos/{vid}/thumbs/{feat_key}/{shot}")
def thumb(vid: str, feat_key: str, shot: int):
    if not feat_key.replace("_", "").isalnum():
        raise HTTPException(400, "bad key")
    p = _cache(vid).path(f"thumbs_{feat_key}") / f"shot_{shot:04d}.jpg"
    if not p.exists():
        raise HTTPException(404, "no thumbnail")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


# ---------------------------------------------------------------- routes: exports

@app.get("/api/videos/{vid}/export.json")
def export_json(vid: str):
    pub = public_result(_current_result(vid))
    return JSONResponse(pub, headers={"Content-Disposition": _attachment(f"{Path(pub['video']).stem}_scenes.json")})


def _attachment(filename: str) -> str:
    """Content-Disposition that survives Arabic/emoji names (HTTP headers must be latin-1):
    ASCII fallback + RFC 5987 UTF-8 filename*."""
    from urllib.parse import quote

    ascii_name = filename.encode("ascii", "ignore").decode().strip() or "scenes.json"
    ascii_name = ascii_name.replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.post("/api/videos/{vid}/export-clips")
def start_clip_export(vid: str):
    result = _current_result(vid)
    meta = _meta(vid)
    out_dir = CFG.paths.exports_dir / vid
    key = ",".join(f"{b:.3f}" for b in result["boundaries"])

    def work(job: Job) -> dict:
        zip_path = out_dir / "clips.zip"
        stamp = out_dir / "clips.key"
        if zip_path.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key:
            return {"zip": str(zip_path)}
        if (out_dir / "clips").exists():
            shutil.rmtree(out_dir / "clips")
        job.stage_label = "Cutting scenes"

        def progress(f):
            job.progress = 0.95 * f

        summary = export_clips(meta["path"], result["scenes"], out_dir / "clips", result["fps"], CFG.export, progress)
        job.stage_label = "Packaging"
        pub = json.dumps(public_result(result), indent=2)
        zip_files([Path(f) for f in summary["files"]], {"scenes.json": pub}, zip_path)
        stamp.write_text(key, encoding="utf-8")
        return {"zip": str(zip_path), **summary}

    return jobs.submit(f"clips-{vid}", "clips", work).public()


@app.get("/api/videos/{vid}/clips.zip")
def clips_zip(vid: str):
    p = CFG.paths.exports_dir / vid / "clips.zip"
    if not p.exists():
        raise HTTPException(404, "clips have not been exported yet")
    stem = Path(_meta(vid)["name"]).stem
    return FileResponse(p, media_type="application/zip", filename=f"{stem}_scenes.zip")


# ---------------------------------------------------------------- routes: corrections & labels

class CorrectionBody(BaseModel):
    op: str                      # merge | split | reset | clear
    scene_index: int | None = None
    at: float | None = None


@app.post("/api/videos/{vid}/corrections")
def correct(vid: str, body: CorrectionBody):
    cache = _cache(vid)
    store = CorrectionStore(cache.dir)
    result = cache.read_json("result.json")
    if result is None:
        raise HTTPException(404, "video has not been analysed yet")
    predicted = result["boundaries"]
    current = (store.load() or {}).get("boundaries", predicted)
    cut_times = [s["start"] for s in result["debug"]["shots"][1:]]
    try:
        if body.op == "reset":
            store.reset()
            return _current_result(vid)
        if body.op == "clear":  # labelling from scratch: one scene, then split by hand
            new, op = [], {"op": "clear"}
        elif body.op == "merge":
            new, op = merge(current, int(body.scene_index)), {"op": "merge", "scene_index": body.scene_index}
        elif body.op == "split":
            new, at = split(current, float(body.at), result["duration"], cut_times)
            op = {"op": "split", "at": at, "requested": body.at}
        else:
            raise HTTPException(400, f"unknown op {body.op}")
    except (CorrectionError, TypeError) as e:
        raise HTTPException(400, str(e))
    store.save(new, op, predicted)
    return _current_result(vid)


class LabelBody(BaseModel):
    annotator: str = ""
    notes: str = ""


@app.post("/api/videos/{vid}/ground-truth")
def save_ground_truth(vid: str, body: LabelBody):
    """Save the current (human-checked) boundaries as a ground-truth label, and place the
    video in data/videos/ so `python -m sceneseen evaluate` can find it."""
    meta = _meta(vid)
    result = _current_result(vid)
    if not result.get("corrected"):
        raise HTTPException(400, "Review the scenes first (split/merge or confirm in edit mode) before saving a label.")
    CFG.paths.videos_dir.mkdir(parents=True, exist_ok=True)
    target = CFG.paths.videos_dir / meta["name"]
    if not target.exists():
        try:
            os.link(meta["path"], target)
        except OSError:
            shutil.copy2(meta["path"], target)
    notes = body.notes or "labelled in SceneSeen UI"
    prov = provenance_from_corrections(CorrectionStore(_cache(vid).dir).load())
    path = save_label(CFG.paths.ground_truth_dir, meta["name"], result["boundaries"], body.annotator, notes, prov)
    return {"saved": str(path), "boundaries": result["boundaries"]}




@app.post("/api/videos/{vid}/confirm")
def confirm(vid: str):
    """Mark the current boundaries as human-reviewed without changing them."""
    cache = _cache(vid)
    result = cache.read_json("result.json")
    store = CorrectionStore(cache.dir)
    current = (store.load() or {}).get("boundaries", result["boundaries"])
    store.save(current, {"op": "confirm"}, result["boundaries"])
    return _current_result(vid)


@app.get("/api/config")
def get_config():
    return {"grouping": CFG.to_dict()["grouping"], "shots": CFG.to_dict()["shots"],
            "features": CFG.to_dict()["features"]}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
