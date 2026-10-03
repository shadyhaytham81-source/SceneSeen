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
