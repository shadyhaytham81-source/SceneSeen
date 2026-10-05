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
from sceneseen.media import VideoError, VideoInfo, ffmpeg_exe, probe, relocated, video_id_for
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
    meta["path"] = str(relocated(meta["path"], CFG.paths.uploads_dir))
    return meta


def _info(vid: str) -> VideoInfo:
    """Stored video info; its path follows the project if the folder was moved."""
    data = _cache(vid).read_json("info.json")
    up = _cache(vid).read_json("upload.json")
    src = up["path"] if up else data["path"]
    data["path"] = str(relocated(src, CFG.paths.uploads_dir))
    return VideoInfo(**data)


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


# ---------------------------------------------------------------- routes: commercial (Phase 2A)

def _commercial(vid: str, allow_inference: bool, progress=None) -> dict:
    """Commercial analysis for the CURRENT scenes. Never raises for model/cache problems."""
    from sceneseen.commercial import analysis as commercial_analysis
    from sceneseen.commercial import review

    result = _current_result(vid)
    info, stage = _info(vid), _stage(vid)
    unique = unique_shots_for(CFG, info, stage, result["scenes"])
    rec = commercial_analysis.analyze(
        stage["cache"].dir, info, stage["shots"], result["scenes"], unique, CFG.commercial,
        clip=stage["features"]["clip"], clip_model=(CFG.features.model, CFG.features.pretrained),
        cache_root=CFG.paths.cache_dir, allow_inference=allow_inference, progress=progress)
    rec["video"] = result["video"]
    rec["frame_base"] = f"/api/videos/{vid}/commercial/frame/"
    return review.attach(rec, review.ReviewStore(CFG.paths.ground_truth_dir, result["video"]).load())


def _commercial_unavailable(e: Exception) -> dict:
    log.error("commercial analysis failed: %s: %s", type(e).__name__, e)
    return {"status": "unavailable", "reason": f"{type(e).__name__}: {e}", "scenes": [], "summary": {}, "notes": []}


@app.get("/api/videos/{vid}/commercial")
def commercial_result(vid: str):
    """Cached commercial results (instant; never loads the detector). status tells the UI whether
    the analysis is ready, partial, not run yet, or unavailable."""
    _meta(vid)
    try:
        return _commercial(vid, allow_inference=False)
    except HTTPException:
        raise
    except Exception as e:   # Phase 1 must keep working whatever happens here
        return _commercial_unavailable(e)


@app.post("/api/videos/{vid}/commercial/analyze")
def start_commercial(vid: str):
    _meta(vid)
    _current_result(vid)

    def work(job: Job) -> dict:
        labels = {"frames": "Preparing representative frames", "detect": "Finding commercial objects"}

        def progress(stage, frac):
            job.stage, job.stage_label = stage, labels.get(stage, "Analysing")
            job.progress = (0.1 * frac) if stage == "frames" else 0.1 + 0.88 * frac

        rec = _commercial(vid, allow_inference=True, progress=progress)
        if rec["status"] == "unavailable":
            raise RuntimeError(rec.get("reason") or "commercial analysis unavailable")
        return {"status": rec["status"], "summary": rec["summary"]}

    return jobs.submit(f"commercial-{vid}", "commercial", work).public()


@app.get("/api/videos/{vid}/commercial/frame/{shot}/{pos}")
def commercial_frame(vid: str, shot: int, pos: int, box: str = "", pad: float = 0.35, size: int = 0):
    """A representative frame, or a padded crop of it (box = x1,y1,x2,y2 as fractions)."""
    import io

    from fastapi.responses import Response
    from PIL import Image

    from sceneseen.commercial.frames import frame_name

    p = _cache(vid).path("commercial") / f"frames_{CFG.commercial.frame_long_side}" / frame_name(shot, pos / 100)
    if not p.exists():
        raise HTTPException(404, "no such frame")
    headers = {"Cache-Control": "max-age=86400"}
    if not box and not size:
        return FileResponse(p, media_type="image/jpeg", headers=headers)
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if box:
                x1, y1, x2, y2 = (float(v) for v in box.split(","))
                w, h = im.size
                bw, bh = (x2 - x1) * w, (y2 - y1) * h
                side = max(bw, bh) * (1 + 2 * pad)
                side = max(side, 0.12 * min(w, h))            # tiny objects still get some context
                cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
                im = im.crop((int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                              int(min(w, cx + side / 2)), int(min(h, cy + side / 2))))
            if size:
                im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
    except (ValueError, OSError):
        raise HTTPException(400, "bad crop request")
    return Response(buf.getvalue(), media_type="image/jpeg", headers=headers)


class ReviewBody(BaseModel):
    key: str | None = None
    verdict: str | None = None       # one of review.VERDICTS, or null to clear
    corrected_type_id: str | None = None   # wrong_label: the right taxonomy type ...
    corrected_label: str | None = None     # ... or free text when the taxonomy does not have it
    missed_label: str | None = None  # report an object SceneSeen did not show
    scene_id: int | None = None
    reviewer: str = ""
    note: str = ""


@app.post("/api/videos/{vid}/commercial/review")
def commercial_review(vid: str, body: ReviewBody):
    from sceneseen.commercial import review

    rec = _commercial(vid, allow_inference=False)
    store = review.ReviewStore(CFG.paths.ground_truth_dir, rec["video"])
    if body.missed_label:
        if body.scene_id is None or not body.missed_label.strip():
            raise HTTPException(400, "missed_label needs a scene_id")
        store.add_missed(body.scene_id, body.missed_label, body.reviewer, body.note)
    else:
        cand = next((c for sc in rec["scenes"] for c in sc["candidates"] if c["key"] == body.key), None)
        if cand is None:
            raise HTTPException(404, "unknown candidate")
        try:
            store.set_verdict(cand, body.verdict, body.reviewer, rec["model"], rec["taxonomy_version"],
                              body.corrected_type_id, body.corrected_label, body.note)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return {"saved": True, "precision": _review_summary()["displayed"]}


def _review_summary() -> dict:
    from sceneseen.commercial import review

    return review.summarize(CFG.paths.ground_truth_dir, None, CFG.commercial.min_relevance, CFG.commercial.min_confidence)


@app.get("/api/commercial/taxonomy")
def commercial_taxonomy():
    """Object types per category (for the "wrong label" correction and the catalogue forms)."""
    from sceneseen.commercial import review
    from sceneseen.commercial import taxonomy as T

    return {"version": T.TAXONOMY_VERSION, "verdicts": review.VERDICTS,
            "categories": [{"id": c, "name": T.CATEGORIES[c].name,
                            "object_types": [{"id": o.id, "label": o.label} for o in T.OBJECTS.values() if o.category == c]}
                           for c in T.CATEGORIES if any(o.category == c for o in T.OBJECTS.values())]}


@app.get("/api/commercial/report")
def commercial_report():
    """Precision of reviewed detections under the current display rules."""
    return _review_summary()


# ---------------------------------------------------------------- routes: catalogue + human product identification (Phase 2B)

_DB = None


def _db():
    """The catalogue database, opened on first use. 503 when it cannot be opened: only the
    catalogue / identification routes are affected, scenes and commercial objects keep working."""
    global _DB
    if _DB is None:
        from sceneseen.catalog.db import Database
        from sceneseen.config import catalog_paths

        try:
            _DB = Database(catalog_paths(CFG)[0])
        except Exception as e:
            log.error("catalogue database unavailable: %s: %s", type(e).__name__, e)
            raise HTTPException(503, f"The product catalogue is unavailable ({type(e).__name__}: {e})")
    return _DB


def _image_store():
    from sceneseen.catalog.images import ImageStore
    from sceneseen.config import catalog_paths

    return ImageStore(catalog_paths(CFG)[1])


from .catalog_api import build_router  # noqa: E402

app.include_router(build_router(_db, _image_store))


def _products(vid: str) -> tuple[dict, dict]:
    """Products per scene for the CURRENT scenes: detected commercial objects + what people
    identified them as. No model runs here and nothing is guessed. If the catalogue database is
    unavailable the objects are still returned, as unidentified."""
    from sceneseen.catalog import identification

    com = _commercial(vid, allow_inference=False)
    out = {"video": com.get("video"), "commercial_status": com["status"], "frame_base": com.get("frame_base"),
           "taxonomy_version": com.get("taxonomy_version"), "catalog": None, "catalog_error": None}
    idents: dict = {}
    try:
        from sceneseen.catalog import service

        db = _db()
        idents = identification.for_video(db, vid)
        out["catalog"] = service.stats(db)
    except HTTPException as e:
        out["catalog_error"] = e.detail
    except Exception as e:
        log.error("reading identifications failed: %s: %s", type(e).__name__, e)
        out["catalog_error"] = f"{type(e).__name__}: {e}"
    out.update(identification.scene_products(com, idents))
    return out, com


@app.get("/api/videos/{vid}/products")
def scene_products(vid: str):
    """Detected objects with their human identifications (instant; loads no model)."""
    _meta(vid)
    return _products(vid)[0]


class IdentifyBody(BaseModel):
    key: str                          # the detected (grouped) object
    actor: str = ""
    brand_id: int | None = None
    product_id: int | None = None
    variant_id: int | None = None
    status: str | None = None         # unknown_product | no_product | not_commercial (or omit when linking)
    confirm: bool = True              # with a product: false = "product identified", not confirmed yet
    notes: str | None = None
    clear: bool = False               # remove the identification (back to unidentified)


@app.post("/api/videos/{vid}/products/identify")
def identify_product(vid: str, body: IdentifyBody):
    """A person identifies one detected object (brand only, exact product, variant, or a
    "no product" status). Every call is appended to the audit history."""
    from sceneseen.catalog import identification

    _meta(vid)
    com = _commercial(vid, allow_inference=False)
    obj = next((c for sc in com.get("scenes", []) for c in sc["candidates"] if c["key"] == body.key), None)
    if obj is None:
        raise HTTPException(404, "unknown object")
    try:
        if body.clear:
            identification.clear(_db(), vid, body.key, body.actor, body.notes)
            return {"saved": True, "identification": None}
        row = identification.identify(_db(), vid, obj, body.actor, body.brand_id, body.product_id, body.variant_id,
                                      body.status, body.confirm, body.notes, com.get("video"))
    except identification.IdentificationError as e:
        raise HTTPException(404 if e.code == "not_found" else 400, e.message)
    return {"saved": True, "identification": row}


@app.get("/api/videos/{vid}/products/history")
def identification_history(vid: str, key: str = ""):
    from sceneseen.catalog import identification

    _meta(vid)
    return {"events": identification.history(_db(), vid, key or None)}


@app.get("/api/videos/{vid}/products/export")
def export_products(vid: str):
    """Final, human-grounded output: per scene, each object with brand / product / status / who."""
    from sceneseen.catalog import identification

    _meta(vid)
    out = _products(vid)[0]
    return {"video": out["video"], "identification_source": "human", "taxonomy": out.get("taxonomy_version"),
            **identification.export_view(out)}


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
