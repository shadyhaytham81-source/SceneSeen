"""EXPERIMENTAL — DISABLED automatic product matching (sceneseen/experimental/matching).

These tests only keep the research code working. The matcher is not part of the SceneSeen
workflow: tests/test_identification.py checks that the app never imports or runs it.
"""
import time

import numpy as np
import pytest
from PIL import Image

from sceneseen.catalog import service as S
from sceneseen.catalog.db import Database, ImageEmbedding
from sceneseen.catalog.images import ImageStore
from sceneseen.commercial import taxonomy as T
from sceneseen.commercial.frames import frame_name
from sceneseen.config import MatchingConfig
from sceneseen.experimental.matching import matcher as M
from sceneseen.experimental.matching.index import CatalogIndex, get_index
from sceneseen.experimental.matching.signals import COLOUR_DIM, COLOUR_KEY, colour_signature, colour_similarity, crop_weight
from sceneseen.experimental.matching.store import CropCache, ensure_catalog_embeddings, missing_catalog_images

from .fakes import FakeEmbedder, jpeg, pattern
from .test_commercial import CFG as CCFG
from .test_commercial import PHONE, WATCH, make_video, run

CFG = MatchingConfig()


@pytest.fixture
def cat(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'catalog.db'}")
    yield db, ImageStore(tmp_path / "images")
    db.dispose()


def add_product(db, store, name, seeds, object_type="smartphone", category="electronics", brand="Acme"):
    brands = {b["name"]: b["id"] for b in S.list_brands(db, include_archived=True)}
    bid = brands.get(brand) or S.create_brand(db, brand)["id"]
    p = S.create_product(db, bid, name=name, category=category, object_type=object_type)
    for s in seeds:
        S.add_image(db, store, p["id"], jpeg(pattern(s)))
    return p["id"]


def video_with(tmp_path, painted: dict):
    """An analysed fake video whose representative frames show `painted` = {shot: [(box, pattern seed)]}."""
    video = make_video(tmp_path)
    cache = video[0]
    fdir = cache / "commercial" / f"frames_{CCFG.frame_long_side}"
    for shot, items in painted.items():
        frame = np.full((360, 640, 3), (shot, 120, 200), np.uint8)       # pixel (0,0) red = shot id (fake detector)
        for box, seed in items:
            x1, y1, x2, y2 = (int(box[0] * 640), int(box[1] * 360), int(box[2] * 640), int(box[3] * 360))
            frame[y1:y2, x1:x2] = np.asarray(Image.fromarray(pattern(seed)).resize((x2 - x1, y2 - y1), Image.NEAREST))
        Image.fromarray(frame).save(fdir / frame_name(shot, 0.5), quality=100, subsampling=0)
    return video, cache / "commercial", fdir


BIG_PHONE = (PHONE[0], 0.80, [0.30, 0.20, 0.70, 0.90])
BIG_WATCH = (WATCH[0], 0.70, [0.05, 0.10, 0.25, 0.60])


def setup_match(tmp_path, cat, phone_seed=1, n_phones=6, extra=None):
    """6 phone products (seeds 1..6); the video shows the phone with pattern `phone_seed`."""
    db, store = cat
    ids = {s: add_product(db, store, f"Phone {s}", [s]) for s in range(1, n_phones + 1)}
    emb = FakeEmbedder()
    ensure_catalog_embeddings(db, store, emb)
    video, cdir, fdir = video_with(tmp_path, {0: [(BIG_PHONE[2], phone_seed)] + (extra or [])})
    com, _ = run(tmp_path, {0: [BIG_PHONE] + ([BIG_WATCH] if extra else [])}, video=video)
    return db, store, emb, com, cdir, fdir, ids


def phone_key(com, type_id="smartphone"):
    return next(c for sc in com["scenes"] for c in sc["candidates"] if c["type_id"] == type_id)["key"]


# ---------------------------------------------------------------- signals

def test_colour_signature_and_similarity():
    red, blue = np.zeros((80, 80, 3), np.uint8), np.zeros((80, 80, 3), np.uint8)
    red[..., 0], blue[..., 2] = 220, 220
    a, b = colour_signature(red), colour_signature(blue)
    assert a.shape == (COLOUR_DIM,) and abs(float(a.sum()) - 1.0) < 1e-5
    sim = colour_similarity(np.stack([a, b]), np.stack([a, b]))
    assert sim[0, 0] > 0.99 and sim[0, 1] < 0.05
    assert crop_weight(400) == 1.0 and crop_weight(20) == 0.25 and 0.25 < crop_weight(90) < 1.0


def test_confidence_model_uses_score_and_lead():
    assert M.match_confidence(1.3, 0.20, CFG) > 0.85                      # clear winner
    assert M.match_confidence(1.3, 0.00, CFG) < CFG.uncertain_confidence  # same score, no lead: a lookalike
    assert M.match_confidence(1.1, 0.02, CFG) < CFG.uncertain_confidence
    assert M.match_state(0.9, CFG) == "high_confidence"
    assert M.match_state(0.9, CFG, cap_at_possible=True) == "possible_match"
    assert M.match_state(0.6, CFG) == "possible_match"
    assert M.match_state(0.3, CFG) == "uncertain"
    assert M.match_state(0.1, CFG) == "no_reliable_match"
    assert CFG.high_confidence > CFG.possible_confidence > CFG.uncertain_confidence > 0


# ---------------------------------------------------------------- embedding cache

def test_catalogue_images_are_embedded_once(cat):
    db, store = cat
    add_product(db, store, "A", [1, 2])
    add_product(db, store, "B", [3])
    emb = FakeEmbedder()
    assert len(missing_catalog_images(db, emb.key)) == 3
    r = ensure_catalog_embeddings(db, store, emb)
    assert r["embedded"] == 3 and emb.images == 3 and r["failed"] == []
    r2 = ensure_catalog_embeddings(db, store, emb)
    assert r2["embedded"] == 0 and emb.images == 3                        # nothing recomputed
    add_product(db, store, "C", [4, 1])                                  # one new picture, one already known content
    assert ensure_catalog_embeddings(db, store, emb)["embedded"] == 1 and emb.images == 4


def test_cache_is_versioned_by_model_and_image_hash(cat):
    db, store = cat
    pid = add_product(db, store, "A", [1])
    v1, v2 = FakeEmbedder("fake:test:v1"), FakeEmbedder("fake:test:v2")
    ensure_catalog_embeddings(db, store, v1)
    assert ensure_catalog_embeddings(db, store, v2)["embedded"] == 1      # new model version: recomputed under its own key
    assert ensure_catalog_embeddings(db, store, v1)["embedded"] == 0      # the old version's cache is still valid
    with db.session() as s:
        keys = {k for (k,) in s.query(ImageEmbedding.model_key)}
    assert keys == {"fake:test:v1", "fake:test:v2", COLOUR_KEY}
    S.add_image(db, store, pid, jpeg(pattern(9)))                         # changed/new image content = new hash
    assert ensure_catalog_embeddings(db, store, v1)["embedded"] == 1


def test_missing_or_corrupt_image_file_is_reported_not_fatal(cat):
    db, store = cat
    add_product(db, store, "A", [1])
    p2 = add_product(db, store, "B", [2])
    img = S.get_product(db, p2)["images"][0]
    store.path(img["sha256"], img["ext"]).unlink()                        # file vanished from disk
    emb = FakeEmbedder()
    r = ensure_catalog_embeddings(db, store, emb)
    assert r["embedded"] == 1 and [f["sha256"] for f in r["failed"]] == [img["sha256"]]
    idx = CatalogIndex(emb.key).build(db)
    assert idx.n_products == 1                                            # the product without a usable image is not matched


def test_crop_cache_round_trip_and_corruption(tmp_path):
    c = CropCache(tmp_path, "m1")
    k = CropCache.key("abc", [0.1, 0.2, 0.3, 0.4])
    c.add([k], np.ones((1, 4), np.float32), np.ones((1, COLOUR_DIM), np.float32), np.array([120.0], np.float32))
    assert CropCache(tmp_path, "m1").get(k)[2] == 120.0
    assert CropCache(tmp_path, "m2").get(k) is None                       # other model: separate file
    c.path.write_bytes(b"broken")
    assert CropCache(tmp_path, "m1").get(k) is None                       # corrupted cache is ignored, not fatal


# ---------------------------------------------------------------- index: gating, ranking, scale

def test_category_gating(cat):
    db, store = cat
    phone = add_product(db, store, "Phone", [1])
    sofa = add_product(db, store, "Sofa", [1], object_type="sofa", category="furniture")        # identical picture!
    armchair = add_product(db, store, "Armchair", [2], object_type="armchair", category="furniture")
    untyped = add_product(db, store, "Some gadget", [3], object_type=None)                      # electronics, no type
    emb = FakeEmbedder()
    ensure_catalog_embeddings(db, store, emb)
    idx = CatalogIndex(emb.key).build(db)
    q, qc, w = emb.embed([pattern(1)]), colour_signature(pattern(1))[None], np.ones(1)
    r = idx.search(q, qc, w, "smartphone", 0.5)
    assert {x["product_id"] for x in r["results"]} == {phone, untyped} and r["results"][0]["product_id"] == phone
    r = idx.search(q, qc, w, "sofa", 0.5)                       # same group (sofa / armchair / chair) is comparable
    assert {x["product_id"] for x in r["results"]} == {sofa, armchair}
    assert idx.search(q, qc, w, "watch", 0.5) == {"eligible_products": 0, "results": []}
    assert idx.search(q, qc, w, "not_a_type", 0.5)["results"] == []
    assert len(idx.search(q, qc, w, "smartphone", 0.5, gate=False)["results"]) == 4


def test_nearest_neighbour_ranking_and_best_reference_image(cat):
    db, store = cat
    ids = [add_product(db, store, f"P{s}", [s]) for s in (1, 2, 3)]
    multi = add_product(db, store, "Multi", [10, 11, 7])        # the third reference image is the matching view
    emb = FakeEmbedder()
    ensure_catalog_embeddings(db, store, emb)
    idx = CatalogIndex(emb.key).build(db)
    q = lambda s: (emb.embed([pattern(s)]), colour_signature(pattern(s))[None], np.ones(1))   # noqa: E731
    r = idx.search(*q(2), "smartphone", 0.5)["results"]
    assert r[0]["product_id"] == ids[1] and r[0]["cosine"] > 0.99 and r[0]["score"] > r[1]["score"]
    assert [x["score"] for x in r] == sorted((x["score"] for x in r), reverse=True)
    r = idx.search(*q(7), "smartphone", 0.5)["results"]
    assert r[0]["product_id"] == multi
    imgs = {i["id"]: i for i in S.get_product(db, multi)["images"]}
    assert imgs[r[0]["image_id"]]["position"] == 2              # reports WHICH reference image matched


def test_archived_products_and_brands_leave_the_index(cat):
    db, store = cat
    a, b = add_product(db, store, "A", [1]), add_product(db, store, "B", [2], brand="Other")
    emb = FakeEmbedder()
    ensure_catalog_embeddings(db, store, emb)
    assert get_index(db, emb.key).n_products == 2
    S.update_product(db, a, archived=True)
    assert get_index(db, emb.key).n_products == 1               # rebuilt automatically when the catalogue changes
    S.update_brand(db, S.get_product(db, b)["brand"]["id"], archived=True)
    assert get_index(db, emb.key).n_products == 0


def synthetic_index(n_products, per_product=3, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    idx = CatalogIndex("synthetic")
    E = rng.standard_normal((n_products * per_product, dim)).astype(np.float32)
    idx.E = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = rng.random((len(E), COLOUR_DIM)).astype(np.float32)
    idx.C = C / C.sum(axis=1, keepdims=True)
    idx.product = np.repeat(np.arange(1, n_products + 1), per_product)
    idx.image = np.arange(len(E))
    types = ["sneakers", "smartphone", "watch", "sofa"]
    idx._type_arr = np.array([types[(p - 1) % 4] for p in idx.product])
    idx._cat_arr = np.array([T.OBJECTS[t].category for t in idx._type_arr])
    idx.types = {int(p): types[(p - 1) % 4] for p in idx.product}
    return idx


def test_large_catalogue_search_is_vectorised_and_exact():
    idx = synthetic_index(10_000)                               # 30,000 reference images
    target = 4242 - (4242 - 1) % 4 + 1                          # a smartphone product
    assert idx.types[target] == "smartphone"
    rows = np.where(idx.product == target)[0]
    q = idx.E[rows[:1]] + 0.05 * np.random.default_rng(1).standard_normal((1, 512)).astype(np.float32)
    q /= np.linalg.norm(q)
    idx.search(q, idx.C[rows[:1]], np.ones(1), "smartphone", 0.5)            # warm-up
    t = time.perf_counter()
    for _ in range(20):
        r = idx.search(q, idx.C[rows[:1]], np.ones(1), "smartphone", 0.5)
    per_object = (time.perf_counter() - t) / 20
    assert r["results"][0]["product_id"] == target and r["eligible_products"] == 2500
    assert per_object < 0.25, f"{per_object * 1000:.1f} ms per object"         # ~5 ms on an M4; generous CI bound
    res = M.rank(idx, q, idx.C[rows[:1]], np.array([300.0]), "smartphone", CFG)
    assert res["state"] == "high_confidence" and res["candidates"][0]["product_id"] == target


# ---------------------------------------------------------------- ranking states

def test_matching_object_is_high_confidence_and_ranked_first(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat, phone_seed=3)
    out = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)
    m = out["matches"][phone_key(com)]
    assert out["status"] == "ready" and m["state"] == "high_confidence"
    assert m["candidates"][0]["product_id"] == ids[3] and m["candidates"][0]["rank"] == 1
    assert m["match_confidence"] >= CFG.high_confidence and m["eligible_products"] == 6
    assert len(m["candidates"]) == CFG.top_k and m["candidates"][1]["match_confidence"] < 0.5
    assert out["summary"]["states"]["high_confidence"] == 1 and out["model"]["embedder_key"] == emb.key


def test_product_not_in_catalogue_is_not_a_confident_match(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat, phone_seed=77)       # 77 is in no product
    m = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)["matches"][phone_key(com)]
    assert m["state"] in ("no_reliable_match", "uncertain")      # the nearest product is NOT presented as the product
    assert m["match_confidence"] < CFG.possible_confidence


def test_lookalikes_are_not_high_confidence(tmp_path, cat):
    db, store = cat
    for i in range(6):                                           # six products with the SAME picture: indistinguishable
        add_product(db, store, f"Clone {i}", [5], brand=f"B{i}")
    emb = FakeEmbedder()
    ensure_catalog_embeddings(db, store, emb)
    video, cdir, fdir = video_with(tmp_path, {0: [(BIG_PHONE[2], 5)]})
    com, _ = run(tmp_path, {0: [BIG_PHONE]}, video=video)
    m = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)["matches"][phone_key(com)]
    assert m["candidates"][0]["cosine"] > 0.75 and m["margin"] < 0.01
    assert m["state"] != "high_confidence"                       # a perfect score without a lead is not a match


def test_small_catalogue_can_never_be_high_confidence(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat, phone_seed=1, n_phones=2)
    m = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)["matches"][phone_key(com)]
    assert m["candidates"][0]["product_id"] == ids[1] and m["candidates"][0]["cosine"] > 0.75
    assert m["state"] == "possible_match" and m["margin_estimated"] and "comparable product" in m["reason"]


def test_no_products_of_that_kind_is_generic_object(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat, extra=[(BIG_WATCH[2], 30)])
    out = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)
    w = out["matches"][phone_key(com, "watch")]
    assert w["state"] == "no_reliable_match" and w["candidates"] == [] and w["eligible_products"] == 0
    assert out["matches"][phone_key(com)]["state"] == "high_confidence"


def test_empty_catalogue(tmp_path, cat):
    db, store = cat
    emb = FakeEmbedder()
    video, cdir, fdir = video_with(tmp_path, {0: [(BIG_PHONE[2], 1)]})
    com, _ = run(tmp_path, {0: [BIG_PHONE]}, video=video)
    out = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)
    assert out["status"] == "ready" and out["matches"][phone_key(com)]["candidates"] == []


# ---------------------------------------------------------------- caching + failures

def test_second_match_uses_cached_crops_and_no_model(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat, phone_seed=2)
    first = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)
    calls = emb.calls
    assert first["summary"]["crops_embedded_now"] >= 1
    cold = FakeEmbedder(fail=True)                               # model unavailable now
    again = M.match_video(com, cdir, fdir, get_index(db, cold.key), cold, CFG, allow_inference=False)
    assert again["status"] == "ready" and again["summary"]["crops_embedded_now"] == 0 and emb.calls == calls
    assert again["matches"][phone_key(com)]["candidates"][0]["product_id"] == ids[2]
    # a catalogue change re-ranks from the cached crops, still without the model
    new = add_product(db, store, "Phone 2 twin", [2], brand="Twin")
    ensure_catalog_embeddings(db, store, emb)
    third = M.match_video(com, cdir, fdir, get_index(db, emb.key), cold, CFG, allow_inference=False)
    assert {c["product_id"] for c in third["matches"][phone_key(com)]["candidates"][:2]} == {ids[2], new}
    assert third["matches"][phone_key(com)]["state"] != "high_confidence"      # now ambiguous: honest downgrade


def test_embedder_failure_degrades_instead_of_raising(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat)
    bad = FakeEmbedder(fail=True)
    out = M.match_video(com, cdir, fdir, get_index(db, bad.key), bad, CFG)
    assert out["status"] == "unavailable" and "offline" in out["reason"] and out["matches"] == {}


def test_not_run_without_inference_and_missing_frames(tmp_path, cat):
    db, store, emb, com, cdir, fdir, ids = setup_match(tmp_path, cat)
    out = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG, allow_inference=False)
    assert out["status"] == "not_run" and emb.calls == 1         # only the catalogue embedding call from the set-up
    for f in fdir.glob("*.jpg"):
        f.unlink()
    out = M.match_video(com, cdir, fdir, get_index(db, emb.key), emb, CFG)
    assert out["status"] in ("unavailable", "partial") and out["matches"] == {}


# ---------------------------------------------------------------- human verification + audit trail
