"""Web API tests against a tiny generated video (no private dataset needed).

Every path (uploads, cache, exports, ground truth, videos) is redirected to a temp folder,
so these tests never touch data/ or ground_truth/.
"""
import importlib
import subprocess

import pytest

from sceneseen.media import ffmpeg_exe

ARABIC_NAME = "مشهد تجريبي ❤️ test.mp4"


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """4 s video: 2 s of one test pattern, a hard cut, 2 s of another (with audio)."""
    d = tmp_path_factory.mktemp("media")
    out = d / ARABIC_NAME
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y",
           "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=2",
           "-f", "lavfi", "-i", "smptebars=size=320x180:rate=25:duration=2",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
           "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return out


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    from fastapi.testclient import TestClient

    root = tmp_path_factory.mktemp("sceneseen")
    cfg = root / "paths.toml"
    cfg.write_text("[paths]\n" + "\n".join(
        f'{k} = "{(root / k).as_posix()}"' for k in
        ("cache_dir", "uploads_dir", "exports_dir", "ground_truth_dir", "videos_dir")) + "\n", encoding="utf-8")
    monkeypatch_module.setenv("SCENESEEN_CONFIG", str(cfg))
    import server.app as app_module

    app_module = importlib.reload(app_module)
    assert str(app_module.CFG.paths.uploads_dir).startswith(str(root))
    return TestClient(app_module.app), app_module


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_frontend_loads(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "SceneSeen" in r.text
    for asset in ("/app.js", "/style.css"):
        assert c.get(asset).status_code == 200


def test_upload_rejects_non_video(client):
    c, _ = client
    r = c.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]
    r = c.post("/api/upload", files={"file": ("fake.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400 and "could not be read" in r.json()["detail"]


def test_unknown_video_is_404(client):
    c, _ = client
    assert c.get("/api/videos/0123456789abcdef/result").status_code == 404


def test_upload_arabic_filename(client, tiny_video):
    c, _ = client
    with open(tiny_video, "rb") as f:
        r = c.post("/api/upload", files={"file": (ARABIC_NAME, f, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == ARABIC_NAME and 3.5 < body["duration"] < 4.5


@pytest.mark.slow
def test_analyze_end_to_end_via_api(client, tiny_video):
    """Full pipeline through the API (downloads CLIP weights on first run)."""
    import time

    c, _ = client
    with open(tiny_video, "rb") as f:
        vid = c.post("/api/upload", files={"file": (ARABIC_NAME, f, "video/mp4")}).json()["video_id"]
    job = c.post(f"/api/videos/{vid}/analyze").json()
    for _ in range(600):
        job = c.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    r = c.get(f"/api/videos/{vid}/result?debug=1").json()
    assert r["video"] == ARABIC_NAME and r["scene_count"] >= 1
    assert r["shot_count"] >= 2                      # the hard cut at 2 s is detected
    uq = c.get(f"/api/videos/{vid}/unique-shots").json()          # post-processing endpoint
    assert uq["summary"]["original_shots"] == r["shot_count"]
    assert sorted(i for sc in uq["scenes"] for g in sc["unique"] for i in g["shot_ids"]) == list(range(r["shot_count"]))
    again = c.get(f"/api/videos/{vid}/result").json()
    assert again["boundaries"] == r["boundaries"]                 # unique shots never change the scenes
    exp = c.get(f"/api/videos/{vid}/export.json")
    assert exp.status_code == 200 and exp.json()["scene_count"] == r["scene_count"]
    assert "filename*=UTF-8''" in exp.headers["content-disposition"]   # Arabic name survives the header


def test_uploads_survive_moving_the_project(client, tiny_video, tmp_path):
    """Caches store absolute paths; after the project folder is moved the stored path is stale.
    The app must find the upload again by file name in the configured uploads folder."""
    import json
    import shutil

    c, app_module = client
    with open(tiny_video, "rb") as f:
        vid = c.post("/api/upload", files={"file": (ARABIC_NAME, f, "video/mp4")}).json()["video_id"]
    cache = app_module.CFG.paths.cache_dir / vid
    meta = json.loads((cache / "upload.json").read_text(encoding="utf-8"))
    real = meta["path"]
    old_home = tmp_path / "old location" / "data" / "uploads"
    meta["path"] = str(old_home / real.replace("\\", "/").split("/")[-1])     # pretend it was recorded elsewhere
    (cache / "upload.json").write_text(json.dumps(meta), encoding="utf-8")
    assert app_module._meta(vid)["path"] == real                              # resolved back by file name
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_relocated_helper(tmp_path):
    from sceneseen.media import relocated

    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "abc.mp4").write_bytes(b"x")
    assert relocated("/gone/elsewhere/abc.mp4", tmp_path / "uploads") == tmp_path / "uploads" / "abc.mp4"
    from pathlib import Path

    assert relocated("/gone/elsewhere/missing.mp4", tmp_path / "uploads") == Path("/gone/elsewhere/missing.mp4")
    assert relocated(tmp_path / "uploads" / "abc.mp4") == tmp_path / "uploads" / "abc.mp4"


# ---------------------------------------------------------------- Phase 2A: commercial endpoints

def _analysed(c, tiny_video):
    import time

    with open(tiny_video, "rb") as f:
        vid = c.post("/api/upload", files={"file": (ARABIC_NAME, f, "video/mp4")}).json()["video_id"]
    if c.get(f"/api/videos/{vid}/result").status_code != 200:
        job = c.post(f"/api/videos/{vid}/analyze").json()
        for _ in range(600):
            job = c.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert job["status"] == "done", job
    return vid


class _PhoneEverywhere:
    """Stands in for the real detector: one confident smartphone per frame."""
    calls = 0

    def detect(self, images, prompts):
        type(self).calls += len(images)
        return [[(prompts.index("smartphone"), 0.83, [0.30, 0.30, 0.60, 0.70])] for _ in images]

    def describe(self):
        return {"name": "fake-detector", "version": "test", "device": "cpu", "load_seconds": 0.0}


def test_commercial_unknown_video_is_404(client):
    c, _ = client
    assert c.get("/api/videos/0123456789abcdef/commercial").status_code == 404


@pytest.mark.slow
def test_commercial_analysis_via_api(client, tiny_video, monkeypatch):
    import time

    from sceneseen.commercial import analysis as commercial_analysis

    c, app_module = client
    vid = _analysed(c, tiny_video)
    scenes_before = c.get(f"/api/videos/{vid}/result").json()["scenes"]

    first = c.get(f"/api/videos/{vid}/commercial").json()
    assert first["status"] == "not_run" and first["summary"]["candidates_shown"] == 0     # nothing runs on GET

    _PhoneEverywhere.calls = 0
    monkeypatch.setattr(commercial_analysis, "get_detector", lambda *a, **k: _PhoneEverywhere())
    job = c.post(f"/api/videos/{vid}/commercial/analyze").json()
    for _ in range(200):
        job = c.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job
    inferred = _PhoneEverywhere.calls
    assert inferred >= 1

    t = time.perf_counter()
    rec = c.get(f"/api/videos/{vid}/commercial").json()
    assert time.perf_counter() - t < 2.0                                   # cached: no model, near instant
    assert _PhoneEverywhere.calls == inferred                              # ... and no new inference
    assert rec["status"] == "ready" and rec["video"] == ARABIC_NAME
    assert rec["summary"]["unique_shots"] == inferred                      # one inference per unique shot
    cands = [x for sc in rec["scenes"] for x in sc["candidates"] if x["displayed"]]
    assert cands and all(x["type_id"] == "smartphone" and x["category_name"] == "Electronics" for x in cands)
    x = cands[0]
    assert x["seen_count"] >= 1 and x["occurrences"] and 0 < x["commercial_relevance"] <= 1
    assert {"context", "categories_present", "candidates"} <= set(rec["scenes"][0])
    assert "box" not in (rec["scenes"][0]["context"].get("venue") or {})   # scene context has no bounding box

    b = x["best"]
    url = f"{rec['frame_base']}{b['shot_id']}/{round(b['position'] * 100)}"
    full = c.get(url)
    crop = c.get(url, params={"box": ",".join(str(v) for v in b["box"]), "size": 200})
    assert full.status_code == 200 and crop.status_code == 200
    assert crop.headers["content-type"] == "image/jpeg" and len(crop.content) < len(full.content)
    assert c.get(f"{rec['frame_base']}9999/50").status_code == 404

    # review: Correct / Wrong / Missed -> precision, stored under the (Arabic) video name
    r = c.post(f"/api/videos/{vid}/commercial/review", json={"key": x["key"], "verdict": "correct", "reviewer": "t"})
    assert r.status_code == 200 and r.json()["precision"]["correct"] == 1
    assert c.post(f"/api/videos/{vid}/commercial/review", json={"key": "nope", "verdict": "wrong"}).status_code == 404
    assert c.post(f"/api/videos/{vid}/commercial/review", json={"key": x["key"], "verdict": "maybe"}).status_code == 400
    c.post(f"/api/videos/{vid}/commercial/review", json={"missed_label": "watch", "scene_id": x["scene_id"], "reviewer": "t"})
    again = c.get(f"/api/videos/{vid}/commercial").json()
    mine = next(y for sc in again["scenes"] for y in sc["candidates"] if y["key"] == x["key"])
    assert mine["review"] == "correct" and again["scenes"][0]["missed"][0]["label"] == "watch"
    rep = c.get("/api/commercial/report").json()
    assert rep["displayed"]["precision"] == 1.0 and rep["missed_reported"] == 1
    stored = list((app_module.CFG.paths.ground_truth_dir / "commercial_reviews").glob("*.json"))
    assert len(stored) == 1 and stored[0].stem in ARABIC_NAME

    # Phase 1 is untouched by all of this
    assert c.get(f"/api/videos/{vid}/result").json()["scenes"] == scenes_before


@pytest.mark.slow
def test_commercial_failure_never_breaks_phase1(client, tiny_video, monkeypatch):
    c, app_module = client
    vid = _analysed(c, tiny_video)

    def explode(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(app_module, "_commercial", explode)
    r = c.get(f"/api/videos/{vid}/commercial")
    assert r.status_code == 200 and r.json()["status"] == "unavailable" and "model exploded" in r.json()["reason"]
    assert c.get(f"/api/videos/{vid}/result").status_code == 200           # scenes still served
    assert c.get(f"/api/videos/{vid}/unique-shots").status_code == 200     # unique shots still served
