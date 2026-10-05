"""Human product identification: the production Phase 2 workflow.

    Video -> Scenes -> Unique Shots -> Commercial Object Detection -> Human Product Identification
          -> Confirmed Scene Products

No model guesses a brand, product, SKU or variant; these tests need no model at all.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from sceneseen.catalog import identification as I
from sceneseen.catalog import service as S
from sceneseen.catalog.db import Database, Identification, IdentificationEvent

from .test_commercial import PHONE, SNEAK, WATCH, make_video, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db(tmp_path):
    d = Database(f"sqlite:///{tmp_path / 'catalog.db'}")
    yield d
    d.dispose()


@pytest.fixture
def world(db):
    """Nike with two sneakers (one with variants), Adidas with one, Apple with a phone."""
    nike, adidas, apple = (S.create_brand(db, n)["id"] for n in ("Nike", "Adidas", "Apple"))
    af1 = S.create_product(db, nike, name="Air Force 1", category="fashion", object_type="sneakers", sku="AF1")["id"]
    dunk = S.create_product(db, nike, name="Dunk Low", category="fashion", object_type="sneakers", sku="DNK")["id"]
    samba = S.create_product(db, adidas, name="Samba", category="fashion", object_type="sneakers", sku="SMB")["id"]
    iphone = S.create_product(db, apple, name="iPhone", category="electronics", object_type="smartphone")["id"]
    white = S.add_variant(db, af1, name="White / 42", color="white", size="42", sku="AF1-W-42")["id"]
    return {"db": db, "nike": nike, "adidas": adidas, "apple": apple, "af1": af1, "dunk": dunk, "samba": samba,
            "iphone": iphone, "white": white}


OBJ = {"key": "obj-sneakers-s08", "type_id": "sneakers", "label": "Sneakers", "scene_id": 8}


def test_identify_brand_only(world):
    db = world["db"]
    r = I.identify(db, "vid", OBJ, "shady", brand_id=world["nike"])
    assert r["status"] == "brand_identified" and r["status_label"] == "Brand identified"
    assert r["brand"]["name"] == "Nike" and r["product"] is None and r["variant"] is None
    assert r["source"] == "human" and r["identified_by"] == "shady" and r["identified_at"]
    assert r["scene_id"] == 8 and r["type_id"] == "sneakers"


def test_identify_exact_product_sets_the_brand_and_confirms(world):
    r = I.identify(world["db"], "vid", OBJ, "shady", product_id=world["af1"])
    assert r["status"] == "confirmed" and r["product"]["name"] == "Air Force 1" and r["product"]["sku"] == "AF1"
    assert r["brand"]["name"] == "Nike"                               # follows the product
    draft = I.identify(world["db"], "vid", {**OBJ, "key": "other"}, "shady", product_id=world["af1"], confirm=False)
    assert draft["status"] == "product_identified"                    # selected, not confirmed yet


def test_identify_variant(world):
    r = I.identify(world["db"], "vid", OBJ, "shady", product_id=world["af1"], variant_id=world["white"], notes="  box-fresh ")
    assert r["variant"] == {"id": world["white"], "name": "White / 42", "color": "white", "size": "42", "sku": "AF1-W-42",
                            "url": None}
    assert r["notes"] == "box-fresh"


@pytest.mark.parametrize("status,label", [("unknown_product", "Unknown product"), ("no_product", "No product"),
                                          ("not_commercial", "Not commercially useful")])
def test_statuses_without_a_product(world, status, label):
    r = I.identify(world["db"], "vid", OBJ, "shady", status=status)
    assert r["status"] == status and r["status_label"] == label and r["brand"] is None and r["product"] is None


def test_rules(world):
    db = world["db"]
    bad = [dict(actor="", brand_id=world["nike"]),                                      # nobody
           dict(actor="s"),                                                             # nothing chosen
           dict(actor="s", brand_id=world["adidas"], product_id=world["af1"]),          # product of another brand
           dict(actor="s", product_id=world["dunk"], variant_id=world["white"]),        # variant of another product
           dict(actor="s", brand_id=world["nike"], variant_id=world["white"]),          # variant without product
           dict(actor="s", brand_id=world["nike"], status="confirmed"),                 # cannot confirm a brand alone
           dict(actor="s", status="no_product", brand_id=world["nike"]),
           dict(actor="s", status="definitely", product_id=world["af1"])]
    for kw in bad:
        with pytest.raises(I.IdentificationError) as e:
            I.identify(db, "vid", OBJ, **kw)
        assert e.value.code == "invalid", kw
    for kw in (dict(product_id=999), dict(brand_id=999), dict(product_id=world["af1"], variant_id=999)):
        with pytest.raises(I.IdentificationError) as e:
            I.identify(db, "vid", OBJ, "s", **kw)
        assert e.value.code == "not_found"
    assert I.for_video(db, "vid") == {} and I.history(db, "vid") == []                  # nothing was written


def test_change_identification_keeps_an_audit_history(world):
    """Nike -> actually Adidas; Nike / unknown -> Nike / Air Force 1. Nothing is silently overwritten."""
    db = world["db"]
    I.identify(db, "vid", OBJ, "shady", brand_id=world["nike"])
    I.identify(db, "vid", OBJ, "shady", product_id=world["af1"])
    I.identify(db, "vid", OBJ, "mona", product_id=world["samba"], notes="three stripes")
    I.identify(db, "vid", OBJ, "mona", status="not_commercial")
    assert I.clear(db, "vid", OBJ["key"], "shady") is True and I.clear(db, "vid", OBJ["key"], "shady") is False
    h = I.history(db, "vid", OBJ["key"])
    assert [(e["action"], e["actor"]) for e in h] == [("clear", "shady"), ("change", "mona"), ("change", "mona"),
                                                      ("change", "shady"), ("identify", "shady")]
    first, to_af1, to_adidas, to_nc, cleared = h[4], h[3], h[2], h[1], h[0]
    assert first["before"] is None and first["after"]["brand"] == "Nike" and first["after"]["product"] is None
    assert to_af1["before"]["status"] == "brand_identified" and to_af1["after"]["product"] == "Air Force 1"
    assert to_adidas["before"]["brand"] == "Nike" and to_adidas["after"]["brand"] == "Adidas" and to_adidas["notes"] == "three stripes"
    assert to_adidas["before"]["identified_by"] == "shady"                              # who had said the previous thing
    assert to_nc["after"]["status"] == "not_commercial" and to_nc["after"]["brand"] is None
    assert cleared["after"] is None and cleared["before"]["status"] == "not_commercial"
    assert all(e["at"] for e in h) and I.for_video(db, "vid") == {}
    with db.session() as s:                                                              # one current row at most, history kept
        assert s.query(Identification).count() == 0 and s.query(IdentificationEvent).count() == 5
    with pytest.raises(I.IdentificationError):
        I.clear(db, "vid", OBJ["key"], " ")


def test_history_survives_renames_and_archiving(world):
    db = world["db"]
    I.identify(db, "vid", OBJ, "shady", product_id=world["af1"])
    S.update_product(db, world["af1"], name="Air Force 1 '07")
    assert S.delete_product(db, world["af1"])["archived"] is True                       # identified -> archived, not deleted
    cur = I.for_video(db, "vid")[OBJ["key"]]
    assert cur["product"]["name"] == "Air Force 1 '07" and cur["product"]["archived"] is True
    assert I.history(db, "vid")[0]["after"]["product"] == "Air Force 1"                 # history shows what was said then


def test_arabic_brand_and_product(db):
    b = S.create_brand(db, "القاهرة للأحذية")["id"]
    p = S.create_product(db, b, name="حذاء رياضي أبيض ⚡", category="fashion", object_type="sneakers", sku="قـ-١")["id"]
    v = S.add_variant(db, p, name="مقاس ٤٢")["id"]
    r = I.identify(db, "فيديو", OBJ, "شادي", product_id=p, variant_id=v, notes="ظهر في المشهد الثامن", video_name="مسلسل ❤️.mp4")
    assert (r["brand"]["name"], r["product"]["name"], r["variant"]["name"]) == ("القاهرة للأحذية", "حذاء رياضي أبيض ⚡", "مقاس ٤٢")
    assert r["identified_by"] == "شادي" and r["notes"] == "ظهر في المشهد الثامن"
    assert I.history(db, "فيديو")[0]["after"]["brand"] == "القاهرة للأحذية"
    assert [x["id"] for x in S.search_products(db, q="رياضي", compatible_with="sneakers")["items"]] == [p]


def test_persistence_across_reopen(tmp_path):
    url = f"sqlite:///{tmp_path / 'c.db'}"
    d = Database(url)
    b = S.create_brand(d, "Nike")["id"]
    I.identify(d, "vid", OBJ, "shady", brand_id=b, notes="swoosh visible")
    d.dispose()
    d2 = Database(url)
    cur = I.for_video(d2, "vid")[OBJ["key"]]
    assert cur["brand"]["name"] == "Nike" and cur["notes"] == "swoosh visible" and len(I.history(d2, "vid")) == 1
    d2.dispose()


def test_create_product_during_identification(world):
    """Product not in the catalogue: create it (brand, name, variant, SKU, URL, category) and link at once."""
    db = world["db"]
    assert S.search_products(db, q="gel kayano", compatible_with="sneakers")["total"] == 0
    brand = S.create_brand(db, "ASICS")["id"]
    p = S.create_product(db, brand, name="Gel-Kayano 14", category="fashion", object_type="sneakers", sku="GK14",
                         url="https://example.com/gk14")
    v = S.add_variant(db, p["id"], name="Silver / 43")
    r = I.identify(db, "vid", OBJ, "shady", product_id=p["id"], variant_id=v["id"])
    assert r["status"] == "confirmed" and r["brand"]["name"] == "ASICS" and r["product"]["url"] == "https://example.com/gk14"
    assert S.search_products(db, q="gk14", compatible_with="sneakers")["items"][0]["id"] == p["id"]


# ---------------------------------------------------------------- repeated objects, scene output, export

def scene_setup(tmp_path):
    """Two scenes. Scene 1: sneakers seen in three camera set-ups (5 shots) + a phone. Scene 2: other sneakers."""
    scenes = [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": 8.0, "shot_start": 0, "shot_end": 3},
              {"scene_id": 2, "start_seconds": 8.0, "end_seconds": 12.0, "shot_start": 4, "shot_end": 5}]
    cache, info, shots, _, _ = make_video(tmp_path, groups=((0, 2), (1, 3), (4, 5)))
    uniq = lambda i, name, g: {"unique_shot_id": name, "representative_shot_id": g[0],     # noqa: E731
                               "occurrences": [{"shot_id": s, "start": shots[s].start, "end": shots[s].end} for s in g]}
    unique = {"scenes": [{"scene_id": 1, "unique": [uniq(0, "S01-U01", (0, 2)), uniq(1, "S01-U02", (1, 3))]},
                         {"scene_id": 2, "unique": [uniq(2, "S02-U01", (4, 5))]}]}
    other = (SNEAK[0], 0.72, [0.10, 0.60, 0.40, 0.90])
    second = [other, WATCH]                 # JPEG may shift the marker pixel of shot 4 by one level
    com, _ = run(tmp_path, {0: [SNEAK, PHONE], 1: [SNEAK], 3: second, 4: second, 5: second},
                 video=(cache, info, shots, scenes, unique))
    return com


def test_repeated_object_inherits_identification_once(tmp_path, world):
    db, com = world["db"], scene_setup(tmp_path)
    s1 = {c["type_id"]: c for c in com["scenes"][0]["candidates"]}
    s2 = {c["type_id"]: c for c in com["scenes"][1]["candidates"]}
    sneakers = s1["sneakers"]
    assert sneakers["seen_count"] == 4 and sneakers["key"] != s2["sneakers"]["key"]       # 4 shots, ONE object
    I.identify(db, "vid", {**sneakers, "scene_id": 1}, "shady", product_id=world["af1"])  # identified once
    out = I.scene_products(com, I.for_video(db, "vid"))
    a = {o["type_id"]: o for o in out["scenes"][0]["objects"]}
    b = {o["type_id"]: o for o in out["scenes"][1]["objects"]}
    assert a["sneakers"]["status"] == "confirmed" and len(a["sneakers"]["occurrences"]) == 4
    exp = I.export_view(out)["scenes"][0]["objects"]
    row = next(o for o in exp if o["type_id"] == "sneakers")
    assert row["product"] == "Air Force 1" and [o["shot_id"] for o in row["occurrences"]] == [0, 1, 2, 3]   # all inherit it
    # ... but NOT the unrelated sneakers of the other scene, nor other objects of this scene
    assert b["sneakers"]["status"] == "unidentified" and b["sneakers"]["identification"] is None
    assert a["smartphone"]["status"] == "unidentified" and b["watch"]["status"] == "unidentified"
    assert out["status_counts"]["confirmed"] == 1 and out["status_counts"]["unidentified"] == 3


def test_final_scene_output_is_human_grounded(tmp_path, world):
    db, com = world["db"], scene_setup(tmp_path)
    s1 = {c["type_id"]: {**c, "scene_id": 1} for c in com["scenes"][0]["candidates"]}
    s2 = {c["type_id"]: {**c, "scene_id": 2} for c in com["scenes"][1]["candidates"]}
    I.identify(db, "vid", s1["sneakers"], "shady", product_id=world["af1"], variant_id=world["white"])
    I.identify(db, "vid", s1["smartphone"], "shady", brand_id=world["apple"])
    I.identify(db, "vid", s2["sneakers"], "mona", status="not_commercial")
    out = I.scene_products(com, I.for_video(db, "vid"))
    exp = {"video": "x.mp4", **I.export_view(out)}
    rows = {(sc["scene_id"], o["type_id"]): o for sc in exp["scenes"] for o in sc["objects"]}
    sneakers, phone, watch, other = rows[(1, "sneakers")], rows[(1, "smartphone")], rows[(2, "watch")], rows[(2, "sneakers")]
    assert (sneakers["brand"], sneakers["product"], sneakers["variant"], sneakers["sku"], sneakers["status"], sneakers["source"]) == (
        "Nike", "Air Force 1", "White / 42", "AF1-W-42", "confirmed", "human")
    assert (phone["brand"], phone["product"], phone["status"], phone["source"]) == ("Apple", None, "brand_identified", "human")
    assert (watch["brand"], watch["product"], watch["status"], watch["source"], watch["identified_by"]) == (None, None, "unidentified", None, None)
    assert other["status"] == "not_commercial" and other["status_label"] == "Not commercially useful"
    # AI confidence only where it makes sense: detection + relevance. No product confidence of any kind.
    for o in rows.values():
        assert 0 < o["detection_confidence"] <= 1 and 0 < o["commercial_relevance"] <= 1
    text = json.dumps({"api": out, "export": exp})
    for banned in ("match_confidence", "match_state", "candidates", "suggestion", "score", "embedder"):
        assert banned not in text, banned
    assert out["scenes"][0]["objects"][0]["commercial_relevance_level"] in ("High", "Medium", "Low")
    # without any catalogue the objects still come out, unidentified
    assert I.scene_products(com, {})["status_counts"]["unidentified"] == 4


def test_identifications_do_not_touch_detection_data(tmp_path, world):
    """Manual catalogue information lives in its own tables, never in the detection cache."""
    db, com = world["db"], scene_setup(tmp_path)
    before = {p: p.read_bytes() for p in (tmp_path / "vid").rglob("*") if p.is_file()}
    c = {**com["scenes"][0]["candidates"][0], "scene_id": 1}
    I.identify(db, "vid", c, "shady", brand_id=world["nike"])
    assert {p: p.read_bytes() for p in (tmp_path / "vid").rglob("*") if p.is_file()} == before
    assert "identification" not in json.dumps(com) and "brand" not in com["scenes"][0]["candidates"][0]


# ---------------------------------------------------------------- the automatic matcher is out of the workflow

def test_automatic_matcher_is_never_loaded_by_the_app_or_the_cli():
    """Importing the server, the CLI and the identification code must not import the experimental
    matcher (so it cannot run, assign products, or add latency)."""
    code = ("import sys; import server.app, sceneseen.__main__, sceneseen.catalog.identification, sceneseen.pipeline, "
            "sceneseen.commercial.analysis; "
            "bad = sorted(m for m in sys.modules if m.startswith('sceneseen.experimental') or m in ('open_clip', 'sklearn')); "
            "print(bad); sys.exit(1 if [m for m in bad if m.startswith('sceneseen.experimental')] else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_workflow_code_references_the_matcher():
    banned = ("experimental", "match_confidence", "match_video", "get_embedder", "suggestion_rank")
    files = [ROOT / "server" / "app.py", ROOT / "server" / "catalog_api.py", ROOT / "sceneseen" / "__main__.py",
             ROOT / "sceneseen" / "pipeline.py", ROOT / "server" / "static" / "catalog.js", ROOT / "server" / "static" / "app.js",
             ROOT / "server" / "static" / "index.html", *(ROOT / "sceneseen" / "catalog").glob("*.py"),
             *(ROOT / "sceneseen" / "commercial").glob("*.py")]
    for f in files:
        text = f.read_text(encoding="utf-8")
        for word in banned:
            if f.name == "db.py" and word == "experimental":
                continue                                    # one comment: the embeddings table belongs to the experiment
            assert word not in text, f"{f.name} mentions {word}"
    api = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
    assert "/products/identify" in api and "/products/match" not in api and "/products/verify" not in api
