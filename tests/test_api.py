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
        ("cache_dir", "uploads_dir", "exports_dir", "ground_truth_dir", "videos_dir")) + "\n"
        + f'[catalog]\ndatabase_url = "sqlite:///{(root / "catalog" / "catalog.db").as_posix()}"\n'
        + f'images_dir = "{(root / "catalog" / "images").as_posix()}"\n', encoding="utf-8")
    (root / "catalog").mkdir()
    monkeypatch_module.delenv("SCENESEEN_DATABASE_URL", raising=False)
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


# ---------------------------------------------------------------- Phase 2B: catalogue API

def _png(seed):
    from .fakes import jpeg, pattern

    return jpeg(pattern(seed), "PNG")


def test_catalog_api_crud_images_and_unicode(client):
    c, app_module = client
    assert str(app_module._db().url).startswith("sqlite:///") and "catalog/catalog.db" in app_module._db().url
    meta = c.get("/api/catalog/meta").json()
    assert {"fashion", "electronics", "accessories"} <= {k["id"] for k in meta["categories"]}
    assert "locations" not in {k["id"] for k in meta["categories"]}          # scene contexts are not product categories
    assert meta["image_roles"][0] == "front"

    b = c.post("/api/catalog/brands", json={"name": "القاهرة للساعات", "website": "https://example.com"})
    assert b.status_code == 200
    assert c.post("/api/catalog/brands", json={"name": "القاهرة للساعات"}).status_code == 409
    assert c.post("/api/catalog/brands", json={"name": " "}).status_code == 400
    assert c.post("/api/catalog/brands", json={"name": "X", "website": "javascript:alert(1)"}).status_code == 400
    bid = b.json()["id"]
    assert c.post("/api/catalog/products", json={"name": "No brand", "category": "accessories"}).status_code == 400
    assert c.post("/api/catalog/products", json={"brand_id": 999, "name": "A", "category": "accessories"}).status_code == 404
    body = {"brand_id": bid, "name": "ساعة يد كلاسيك", "category": "accessories", "object_type": "watch", "sku": "QW-1",
            "url": "https://example.com/qw1", "price": 2450, "currency": "egp"}
    p = c.post("/api/catalog/products", json=body)
    assert p.status_code == 200 and p.json()["currency"] == "EGP" and p.json()["object_label"] == "Watch"
    pid = p.json()["id"]
    assert c.post("/api/catalog/products", json=body).status_code == 409                  # duplicate product
    assert c.post("/api/catalog/products", json={**body, "name": "B", "sku": "Q2", "url": "ftp://x"}).status_code == 400

    up = c.post(f"/api/catalog/products/{pid}/images", data={"role": "front"},
                files=[("files", ("front.png", _png(1), "image/png")), ("files", ("back.png", _png(2), "image/png")),
                       ("files", ("broken.jpg", b"not an image", "image/jpeg"))])
    assert up.status_code == 200 and len(up.json()["saved"]) == 2 and up.json()["failed"][0]["filename"] == "broken.jpg"
    only_bad = c.post(f"/api/catalog/products/{pid}/images", files=[("files", ("x.jpg", b"zzz", "image/jpeg"))])
    assert only_bad.status_code == 400
    assert c.post("/api/catalog/products/999/images", files=[("files", ("a.png", _png(3), "image/png"))]).status_code == 404
    got = c.get(f"/api/catalog/products/{pid}").json()
    assert got["image_count"] == 2 and got["name"] == "ساعة يد كلاسيك"
    img = got["images"][1]
    assert c.patch(f"/api/catalog/images/{img['id']}", json={"role": "back"}).json()["role"] == "back"
    full, thumb = c.get(img["url"]), c.get(img["url"], params={"size": 64})
    assert full.status_code == 200 and thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert c.get("/api/catalog/images/" + "0" * 64 + ".jpg").status_code == 404
    assert c.get("/api/catalog/images/..%2f..%2fsecret.jpg").status_code in (400, 404)

    found = c.get("/api/catalog/products", params={"q": "ساعة", "category": "accessories"}).json()
    assert found["total"] == 1 and found["items"][0]["primary_image"]["url"].startswith("/api/catalog/images/")
    assert c.get("/api/catalog/products", params={"category": "fashion"}).json()["total"] == 0
    assert c.patch(f"/api/catalog/products/{pid}", json={"name": "ساعة ٢", "price": None}).json()["name"] == "ساعة ٢"
    v = c.post(f"/api/catalog/products/{pid}/variants", json={"name": "Black", "color": "black"}).json()
    assert c.delete(f"/api/catalog/variants/{v['id']}").json() == {"deleted": True}
    assert c.delete(f"/api/catalog/images/{img['id']}").json() == {"deleted": True}
    st = c.get("/api/catalog/status").json()
    assert st["products"] == 1 and st["images_pending"] == 1 and st["embedder_loaded"] is False
    assert c.patch(f"/api/catalog/products/{pid}", json={"archived": True}).json()["archived"] is True
    assert c.get("/api/catalog/products").json()["total"] == 0
    assert c.get("/api/catalog/products", params={"archived": True}).json()["total"] == 1
    assert c.delete(f"/api/catalog/products/{pid}").json()["deleted"] is True
    assert c.get(f"/api/catalog/products/{pid}").status_code == 404
    assert c.get("/api/commercial/taxonomy").json()["verdicts"]["wrong_label"]


def test_catalog_unavailable_is_503_and_everything_else_works(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "_DB", None)
    monkeypatch.setenv("SCENESEEN_DATABASE_URL", "sqlite:////nonexistent-folder/deeper/catalog.db")
    r = c.get("/api/catalog/products")
    assert r.status_code == 503 and "catalogue is unavailable" in r.json()["detail"]
    assert c.get("/api/catalog/meta").status_code == 503
    assert c.get("/").status_code == 200 and c.get("/api/videos").status_code == 200
    monkeypatch.delenv("SCENESEEN_DATABASE_URL")
    monkeypatch.setattr(app_module, "_DB", None)
    assert c.get("/api/catalog/products").status_code == 200                # recovers without a restart


def _wait(c, job):
    import time

    for _ in range(300):
        job = c.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError(job)


@pytest.mark.slow
def test_products_end_to_end_via_api(client, tiny_video, monkeypatch):
    """Acceptance flow: brand -> products + images -> analysed video -> objects -> match -> confirm /
    change / no match -> final output -> reload persists -> cached and fast."""
    import time

    import numpy as np
    from PIL import Image

    from sceneseen.commercial import analysis as commercial_analysis
    from sceneseen.commercial.frames import load_rgb

    from .fakes import FakeEmbedder, jpeg, pattern

    c, app_module = client
    vid = _analysed(c, tiny_video)
    scenes_before = c.get(f"/api/videos/{vid}/result").json()["scenes"]
    monkeypatch.setattr(commercial_analysis, "get_detector", lambda *a, **k: _PhoneEverywhere())
    fake = FakeEmbedder()
    monkeypatch.setattr(app_module, "_embedder", lambda: fake)
    if c.get(f"/api/videos/{vid}/commercial").json()["status"] != "ready":
        assert _wait(c, c.post(f"/api/videos/{vid}/commercial/analyze").json())["status"] == "done"
    com = c.get(f"/api/videos/{vid}/commercial").json()
    objs = [x for sc in com["scenes"] for x in sc["candidates"] if x["displayed"]]
    assert objs

    before = c.get(f"/api/videos/{vid}/products").json()                    # nothing matched yet, objects still listed
    assert before["matching"]["status"] == "not_run" and before["status_counts"]["not_matched"] == len(objs)

    # catalogue: the phone that is really in the video (cut from its frame) + 6 other phones + 1 sofa
    brand = c.post("/api/catalog/brands", json={"name": "Delta Tech"}).json()["id"]
    x = objs[0]
    frames = app_module._cache(vid).path("commercial") / f"frames_{app_module.CFG.commercial.frame_long_side}"
    frame = load_rgb(next(frames.glob(f"shot_{x['best']['shot_id']:04d}_*.jpg")))
    h, w = frame.shape[:2]
    b = x["best"]["box"]
    real = np.ascontiguousarray(frame[int(b[1] * h):int(b[3] * h), int(b[0] * w):int(b[2] * w)])
    real = np.asarray(Image.fromarray(real).resize((240, 240)))
    ids = {}
    for name, img in [("Real phone", real)] + [(f"Other phone {i}", pattern(40 + i)) for i in range(6)]:
        pid = c.post("/api/catalog/products", json={"brand_id": brand, "name": name, "category": "electronics",
                                                    "object_type": "smartphone"}).json()["id"]
        assert c.post(f"/api/catalog/products/{pid}/images", files=[("files", (f"{name}.jpg", jpeg(img), "image/jpeg"))]).status_code == 200
        ids[name] = pid
    assert c.get("/api/catalog/status").json()["images_pending"] == 7

    job = _wait(c, c.post(f"/api/videos/{vid}/products/match").json())
    assert job["status"] == "done", job
    embedded_images = fake.images
    assert c.get("/api/catalog/status").json()["images_pending"] == 0

    t = time.perf_counter()
    out = c.get(f"/api/videos/{vid}/products").json()
    assert time.perf_counter() - t < 2.0 and fake.images == embedded_images      # cached: fast, no model call
    assert out["matching"]["status"] == "ready" and out["matching"]["summary"]["catalogue_products"] == 7
    obj = next(o for sc in out["scenes"] for o in sc["objects"] if o["key"] == x["key"])
    assert obj["candidates"][0]["product"]["name"] == "Real phone"               # ranked first
    assert obj["status"] in ("high_confidence_candidate", "needs_review")        # honest state, never auto-confirmed
    assert obj["verification_status"] == "unverified" and obj["product"] is None
    for f in ("detection_confidence", "commercial_relevance", "match_confidence"):
        assert 0 <= obj[f] <= 1

    # human decisions
    url = f"/api/videos/{vid}/products/verify"
    assert c.post(url, json={"key": x["key"], "action": "confirm", "product_id": ids["Real phone"]}).status_code == 400   # who?
    assert c.post(url, json={"key": "nope", "action": "no_match", "actor": "t"}).status_code == 404
    assert c.post(url, json={"key": x["key"], "action": "confirm", "product_id": 99999, "actor": "t"}).status_code == 400
    r = c.post(url, json={"key": x["key"], "action": "confirm", "product_id": ids["Real phone"], "actor": "shady"}).json()
    assert r["verification"]["status"] == "confirmed" and r["verification"]["suggestion_rank"] == 1
    assert r["verification"]["models"]["embedder"] == fake.key and r["verification"]["models"]["calibration"]
    r = c.post(url, json={"key": x["key"], "action": "confirm", "product_id": ids["Other phone 2"], "actor": "mona",
                          "source": "search"}).json()
    assert r["verification"]["product"]["name"] == "Other phone 2"

    # reload: decisions persist (new database handle = what a server restart does)
    monkeypatch.setattr(app_module, "_DB", None)
    out = c.get(f"/api/videos/{vid}/products").json()
    obj = next(o for sc in out["scenes"] for o in sc["objects"] if o["key"] == x["key"])
    assert obj["status"] == "confirmed" and obj["product"]["name"] == "Other phone 2" and obj["verification"]["decided_by"] == "mona"
    assert obj["match_confidence"] is not None and obj["detection_confidence"] == x["detection_confidence"]   # signals kept
    hist = c.get(f"/api/videos/{vid}/products/history", params={"key": x["key"]}).json()["events"]
    assert [e["action"] for e in hist] == ["change", "confirm"] and hist[0]["previous_product_id"] == ids["Real phone"]
    exp = c.get(f"/api/videos/{vid}/products/export").json()
    row = next(o for sc in exp["scenes"] for o in sc["objects"] if o["product"])
    assert row["product"] == "Other phone 2" and row["brand"] == "Delta Tech" and row["verified_by"] == "mona"
    assert exp["video"] == ARABIC_NAME and "start_seconds" in exp["scenes"][0]

    # a confirmed product cannot silently disappear
    assert c.delete(f"/api/catalog/products/{ids['Other phone 2']}").json()["archived"] is True
    r = c.post(url, json={"key": x["key"], "action": "no_match", "actor": "shady"}).json()
    assert r["verification"]["status"] == "no_match"
    out = c.get(f"/api/videos/{vid}/products").json()
    assert next(o for sc in out["scenes"] for o in sc["objects"] if o["key"] == x["key"])["status"] == "no_match_confirmed"
    assert c.post(url, json={"key": x["key"], "action": "clear", "actor": "shady"}).json()["verification"] is None

    # Phase 1 is untouched by all of this
    assert c.get(f"/api/videos/{vid}/result").json()["scenes"] == scenes_before


@pytest.mark.slow
def test_matching_failures_never_break_objects_or_scenes(client, tiny_video, monkeypatch):
    from sceneseen.commercial import analysis as commercial_analysis

    from .fakes import FakeEmbedder

    c, app_module = client
    vid = _analysed(c, tiny_video)
    monkeypatch.setattr(commercial_analysis, "get_detector", lambda *a, **k: _PhoneEverywhere())
    if c.get(f"/api/videos/{vid}/commercial").json()["status"] != "ready":
        assert _wait(c, c.post(f"/api/videos/{vid}/commercial/analyze").json())["status"] == "done"
    n = sum(len(sc["candidates"]) for sc in c.get(f"/api/videos/{vid}/commercial").json()["scenes"])

    # 1. embedding model cannot be loaded (download failed): job reports it, objects still listed, retry possible
    monkeypatch.setattr(app_module, "_embedder", lambda: FakeEmbedder("fake:other:v9", fail=True))
    job = _wait(c, c.post(f"/api/videos/{vid}/products/match").json())
    assert job["status"] == "error" and "offline" in job["error"]
    out = c.get(f"/api/videos/{vid}/products")
    assert out.status_code == 200 and sum(len(sc["objects"]) for sc in out.json()["scenes"]) >= 1
    assert out.json()["matching"]["status"] in ("not_run", "unavailable")
    monkeypatch.setattr(app_module, "_embedder", lambda: FakeEmbedder("fake:other:v9"))
    assert _wait(c, c.post(f"/api/videos/{vid}/products/match").json())["status"] == "done"       # retry works

    # 2. catalogue database unavailable: objects come back as generic, commercial + scenes untouched
    monkeypatch.setattr(app_module, "_DB", None)
    monkeypatch.setenv("SCENESEEN_DATABASE_URL", "sqlite:////nonexistent-folder/deeper/catalog.db")
    out = c.get(f"/api/videos/{vid}/products").json()
    assert out["matching"]["status"] == "unavailable" and out["catalog"] is None
    assert all(o["status"] == "not_matched" and o["product"] is None for sc in out["scenes"] for o in sc["objects"])
    assert sum(len(sc["candidates"]) for sc in c.get(f"/api/videos/{vid}/commercial").json()["scenes"]) == n
    assert c.get(f"/api/videos/{vid}/result").status_code == 200
    monkeypatch.delenv("SCENESEEN_DATABASE_URL")
    monkeypatch.setattr(app_module, "_DB", None)

    # 3. the matcher itself explodes
    from sceneseen.matching import matcher

    monkeypatch.setattr(matcher, "match_video", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("matcher exploded")))
    out = c.get(f"/api/videos/{vid}/products").json()
    assert out["matching"]["status"] == "unavailable" and "matcher exploded" in out["matching"]["reason"]
    assert sum(len(sc["objects"]) for sc in out["scenes"]) >= 1
