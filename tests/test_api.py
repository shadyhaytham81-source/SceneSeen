"""Web API tests against a tiny generated video (no private dataset needed).

Every path (uploads, cache, exports, ground truth, videos) is redirected to a temp folder,
so these tests never touch data/ or ground_truth/.
"""
import importlib
import json
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
    assert c.get("/api/catalog/status").status_code in (404, 405)           # the matcher's endpoints are gone
    assert c.post("/api/catalog/embed").status_code in (404, 405)
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
def test_human_identification_end_to_end_via_api(client, tiny_video, monkeypatch):
    """Video -> scenes -> unique shots -> commercial objects -> a PERSON identifies the product ->
    confirmed scene products. No embedding model, no matcher, no product confidence anywhere."""
    import time

    from sceneseen.commercial import analysis as commercial_analysis

    from .fakes import jpeg, pattern

    from sceneseen.experimental.matching import embedder, matcher

    c, app_module = client
    ran = []
    monkeypatch.setattr(matcher, "match_video", lambda *a, **k: ran.append("match_video"))
    monkeypatch.setattr(matcher, "rank", lambda *a, **k: ran.append("rank"))
    monkeypatch.setattr(embedder, "get_embedder", lambda *a, **k: ran.append("get_embedder"))
    monkeypatch.setattr(embedder.Embedder, "load", lambda self: ran.append("load"))
    vid = _analysed(c, tiny_video)
    scenes_before = c.get(f"/api/videos/{vid}/result").json()["scenes"]
    monkeypatch.setattr(commercial_analysis, "get_detector", lambda *a, **k: _PhoneEverywhere())
    if c.get(f"/api/videos/{vid}/commercial").json()["status"] != "ready":
        assert _wait(c, c.post(f"/api/videos/{vid}/commercial/analyze").json())["status"] == "done"
    com = c.get(f"/api/videos/{vid}/commercial").json()
    objs = [x for sc in com["scenes"] for x in sc["candidates"] if x["displayed"]]
    x = objs[0]
    url = f"/api/videos/{vid}/products/identify"

    # nothing is guessed: every object starts unidentified, with only the two detection signals
    out = c.get(f"/api/videos/{vid}/products").json()
    flat = [o for sc in out["scenes"] for o in sc["objects"]]
    assert len(flat) == len(objs) and all(o["status"] == "unidentified" and o["identification"] is None for o in flat)
    text = json.dumps(out)
    for banned in ("match_confidence", "candidates", "match_state", "suggestion", "embedder"):
        assert banned not in text, banned
    assert 0 < flat[0]["detection_confidence"] <= 1 and flat[0]["commercial_relevance_level"] in ("High", "Medium", "Low")
    for gone in ("/products/match", "/products/verify"):
        assert c.post(f"/api/videos/{vid}{gone}", json={}).status_code in (404, 405)

    # catalogue (Arabic brand) and the filtered search a person uses
    brand = c.post("/api/catalog/brands", json={"name": "دلتا تك"}).json()["id"]
    phone = c.post("/api/catalog/products", json={"brand_id": brand, "name": "هاتف دلتا ١٢", "category": "electronics",
                                                  "object_type": "smartphone", "sku": "DT-12"}).json()
    variant = c.post(f"/api/catalog/products/{phone['id']}/variants", json={"name": "أسود 256", "sku": "DT-12-BLK"}).json()
    sofa = c.post("/api/catalog/products", json={"brand_id": brand, "name": "Sofa", "category": "furniture",
                                                 "object_type": "sofa"}).json()
    found = c.get("/api/catalog/products", params={"compatible_with": x["type_id"]}).json()
    assert [p["id"] for p in found["items"]] == [phone["id"]]                       # the sofa is filtered out
    assert c.get("/api/catalog/products", params={"q": "dt-12-blk"}).json()["items"][0]["id"] == phone["id"]
    assert c.get("/api/catalog/products", params={"compatible_with": "nonsense"}).status_code == 400
    assert c.get("/api/catalog/brands", params={"q": "دلتا", "compatible_with": x["type_id"]}).json()["brands"][0]["product_count"] == 1

    # rules
    assert c.post(url, json={"key": x["key"], "brand_id": brand}).status_code == 400                        # who?
    assert c.post(url, json={"key": "nope", "brand_id": brand, "actor": "t"}).status_code == 404
    assert c.post(url, json={"key": x["key"], "product_id": 99999, "actor": "t"}).status_code == 404
    assert c.post(url, json={"key": x["key"], "actor": "t"}).status_code == 400                             # nothing chosen
    assert c.post(url, json={"key": x["key"], "actor": "t", "status": "no_product", "brand_id": brand}).status_code == 400

    # brand only -> exact product -> variant -> different product: every step is kept in the history
    r = c.post(url, json={"key": x["key"], "brand_id": brand, "actor": "shady"}).json()["identification"]
    assert r["status"] == "brand_identified" and r["brand"]["name"] == "دلتا تك" and r["product"] is None and r["source"] == "human"
    r = c.post(url, json={"key": x["key"], "product_id": phone["id"], "actor": "shady", "confirm": False}).json()["identification"]
    assert r["status"] == "product_identified"
    r = c.post(url, json={"key": x["key"], "product_id": phone["id"], "variant_id": variant["id"], "actor": "mona",
                          "notes": "شاشة مكسورة"}).json()["identification"]
    assert r["status"] == "confirmed" and r["variant"]["name"] == "أسود 256" and r["notes"] == "شاشة مكسورة"

    # product not in the catalogue: create it (with an image) and link it straight away
    new = c.post("/api/catalog/products", json={"brand_id": brand, "name": "Delta Fold", "category": "electronics",
                                                "object_type": "smartphone"}).json()
    assert c.post(f"/api/catalog/products/{new['id']}/images", files=[("files", ("f.jpg", jpeg(pattern(3)), "image/jpeg"))]).status_code == 200
    r = c.post(url, json={"key": x["key"], "product_id": new["id"], "actor": "mona"}).json()["identification"]
    assert r["product"]["name"] == "Delta Fold" and r["variant"] is None and r["product"]["image"]

    # reload: persisted (a new database handle is what a server restart does); other objects did not inherit it
    monkeypatch.setattr(app_module, "_DB", None)
    t = time.perf_counter()
    out = c.get(f"/api/videos/{vid}/products").json()
    assert time.perf_counter() - t < 2.0
    flat = [o for sc in out["scenes"] for o in sc["objects"]]
    mine = next(o for o in flat if o["key"] == x["key"])
    assert mine["status"] == "confirmed" and mine["identification"]["identified_by"] == "mona"
    assert mine["detection_confidence"] == x["detection_confidence"]
    assert all(o["status"] == "unidentified" for o in flat if o["key"] != x["key"])       # same label elsewhere: untouched
    assert len(mine["occurrences"]) == x["seen_count"]                                    # one identification, every occurrence
    hist = c.get(f"/api/videos/{vid}/products/history", params={"key": x["key"]}).json()["events"]
    assert [e["action"] for e in hist] == ["change", "change", "change", "identify"]
    assert hist[0]["before"]["product"] == "هاتف دلتا ١٢" and hist[0]["after"]["product"] == "Delta Fold"
    assert hist[-1]["after"]["status"] == "brand_identified" and hist[-1]["before"] is None

    exp = c.get(f"/api/videos/{vid}/products/export").json()
    row = next(o for sc in exp["scenes"] for o in sc["objects"] if o["object_id"] == x["key"])
    assert (row["brand"], row["product"], row["status"], row["source"], row["identified_by"]) == ("دلتا تك", "Delta Fold", "confirmed", "human", "mona")
    assert exp["identification_source"] == "human" and exp["video"] == ARABIC_NAME and "match_confidence" not in json.dumps(exp)

    # the other statuses, and removing an identification
    for st in ("unknown_product", "no_product", "not_commercial"):
        r = c.post(url, json={"key": x["key"], "status": st, "actor": "shady"}).json()["identification"]
        assert r["status"] == st and r["brand"] is None and r["product"] is None
    assert c.post(url, json={"key": x["key"], "clear": True, "actor": "shady"}).json()["identification"] is None
    out = c.get(f"/api/videos/{vid}/products").json()
    assert next(o for sc in out["scenes"] for o in sc["objects"] if o["key"] == x["key"])["status"] == "unidentified"
    assert c.get(f"/api/videos/{vid}/products/history", params={"key": x["key"]}).json()["events"][0]["action"] == "clear"
    assert c.delete(f"/api/catalog/products/{sofa['id']}").json()["deleted"] is True

    # the experimental matcher was never run by any of this (it would have exploded, see the top of the test)
    assert not ran
    assert c.get(f"/api/videos/{vid}/result").json()["scenes"] == scenes_before           # Phase 1 untouched


@pytest.mark.slow
def test_catalogue_failure_never_breaks_objects_or_scenes(client, tiny_video, monkeypatch):
    from sceneseen.commercial import analysis as commercial_analysis

    c, app_module = client
    vid = _analysed(c, tiny_video)
    monkeypatch.setattr(commercial_analysis, "get_detector", lambda *a, **k: _PhoneEverywhere())
    if c.get(f"/api/videos/{vid}/commercial").json()["status"] != "ready":
        assert _wait(c, c.post(f"/api/videos/{vid}/commercial/analyze").json())["status"] == "done"
    n = sum(len(sc["candidates"]) for sc in c.get(f"/api/videos/{vid}/commercial").json()["scenes"])
    monkeypatch.setattr(app_module, "_DB", None)
    monkeypatch.setenv("SCENESEEN_DATABASE_URL", "sqlite:////nonexistent-folder/deeper/catalog.db")
    out = c.get(f"/api/videos/{vid}/products")
    assert out.status_code == 200 and out.json()["catalog"] is None and "unavailable" in out.json()["catalog_error"]
    flat = [o for sc in out.json()["scenes"] for o in sc["objects"]]
    assert flat and all(o["status"] == "unidentified" for o in flat)                 # objects still listed
    assert c.post(f"/api/videos/{vid}/products/identify", json={"key": flat[0]["key"], "status": "no_product",
                                                               "actor": "t"}).status_code == 503
    assert sum(len(sc["candidates"]) for sc in c.get(f"/api/videos/{vid}/commercial").json()["scenes"]) == n
    assert c.get(f"/api/videos/{vid}/result").status_code == 200
    assert c.get(f"/api/videos/{vid}/products/export").status_code == 200
