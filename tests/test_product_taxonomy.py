"""Phase 2B product matching: taxonomy normalisation, type filtering, colour, ranking, index
persistence and missing / incompatible data. A fake encoder replaces OpenCLIP (no model download)."""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sceneseen.commercial import matching as M
from sceneseen.commercial.product_taxonomy import ProductInfo, load_taxonomy

T = load_taxonomy()


# ---------------------------------------------------------------- taxonomy: catalogue products

@pytest.mark.parametrize("name,ptype,expected", [
    # real IN YOUR SHOE names / product types (shop jargon included)
    ("Cairo Is A Mindset Oversized Hoodie", "Printed Hoodies", "hoodie"),
    ("Black Basic Oversized Crewneck", "Crewnecks", "sweatshirt"),
    ("Game Night Pjoys", "PJOYS", "pajama_set"),
    ("Black ZED Stars Wide Leg Sweatpants", "Swants", "sweatpants"),
    ("Green ZED Stars Sworts", "Sworts", "sweat_shorts"),
    ("Female Off White Wide Leg Jorts", "Jorts", "denim_shorts"),
    ("Happy Day Swim Shorts", "Swim Shorts", "swim_shorts"),
    ("Garden Neck Socks", "Neck Socks", "socks"),
    ("White Core Short Socks", "Short Socks", "socks"),               # "short" is not "shorts"
    ("Black Short Sleeve Shirt", "Short Sleeve Shirt", "shirt"),     # ... and a short-sleeve shirt is a shirt
    ("Black Overshirt Jacket", "Overshirt Jacket", "overshirt"),     # overshirt before shirt / jacket
    ("Blue Narrow Striped Boxy Polo", "Polo T-Shirts", "polo"),      # polo before t-shirt
    ("Cinnamon Corduroy Jacket", "Curduroy Jackets", "jacket"),      # the shop's own typo
    ("Female Butter Yellow Oversized Linen Set", "Linen Set", "set"),
    ("Female Dark Blue Striped Boxer Pants", "Boxer Pants", "lounge_pants"),
    ("Male Grey Loose Office Pants", "Pants", "trousers"),
    ("Orange Ribbed Tank Top", "Tops", "top"),                        # product_type wins over the name
    ("Black Washed Cowboy Bucket Hat", "Bucket", "bucket_hat"),
    ("Black Plaid Tote Bag", "Wool Tote Bag", "tote_bag"),
    ("Coconut Scented Candle 100 ML", "Candle", "candle"),
    ("Some Tee", "", "tshirt"),                                       # no product_type: the name is used
])
def test_product_types_from_real_catalogue_names(name, ptype, expected):
    info = T.describe_product(name, ptype)
    assert info.product_type == expected
    assert info.umbrella == T.types[expected]["umbrella"]


def test_ambiguous_products_stay_unknown():
    for name, ptype in [("Dusty Pink Checks Maxi Flowy Wrap", "Flowy Wraps Maxi"), ("Lime Green Plaid Mini Wrap", "Flowy Wraps"),
                        ("Mystery Thing", "Gift Card"), ("", "")]:
        info = T.describe_product(name, ptype)
        assert info.product_type is None and info.umbrella is None and info.type_source == ""


def test_original_name_is_not_needed_back_and_type_source_is_recorded():
    assert T.describe_product("Black Basic Oversized Crewneck", "Crewnecks").type_source == "product_type"
    assert T.describe_product("Navy Hoodie", "").type_source == "name"


@pytest.mark.parametrize("name,colors", [
    ("Black Basic Oversized Hoodie", ("black",)),
    ("Female Off White Wide Leg Jorts", ("white",)),                  # "off white" is one colour, not "white"+"off"
    ("Camel Brown Boyfriend Linen Shirt", ("brown",)),
    ("Navy Playground Neck Socks", ("blue",)),
    ("Olive Washed Cap", ("green",)),
    ("Burgundy & Pink Striped Daydreamer Boxy Long Sleeves", ("red", "pink")),
    ("Cairo Is A Mindset Oversized Hoodie", ()),                      # printed design: colour unknown
    ("Cereal Killer Pjoys", ()),                                      # motif words are not colours
])
def test_colours_from_names(name, colors):
    assert T.describe_product(name, "Tops").colors == colors


def test_colour_options_and_patterns():
    info = T.describe_product("Cereal Killer Pjoys", "PJOYS", ["Black", "White"])
    assert info.colors == ("black", "white") and info.multicolor
    assert T.describe_product("Red Grid Boxy Linen Shirt", "Boxy Linen Shirts").patterned
    assert not T.describe_product("Black Basic Oversized Hoodie", "Plain Hoodies").patterned


# ---------------------------------------------------------------- taxonomy: detections

def test_owlv2_detection_normalisation():
    q = T.detection_query("Black jacket", "jacket", "black")
    assert q.detection_type == "jacket" and q.color == "black" and q.umbrella == "outerwear"
    assert "jacket" in q.strict_types and "overshirt" in q.strict_types and "tshirt" not in q.strict_types
    assert "hoodie" in q.broad_types and "trousers" not in q.strict_types + q.broad_types
    q = T.detection_query("Jacket", "jacket", None)
    assert q.color is None                                            # no colour invented
    q = T.detection_query("Trousers", "trousers", None)
    assert "jeans" in q.strict_types and "shorts" in q.broad_types and "shirt" not in q.strict_types + q.broad_types


def test_free_text_labels():
    q = T.detection_query("black pants")
    assert q.detection_type == "trousers" and q.color == "black" and q.notes
    q = T.detection_query("grey hoodie")
    assert q.strict_types == ("hoodie",) and "sweatshirt" in q.broad_types and q.color == "grey"
    q = T.detection_query("red and blue shirt")
    assert q.color is None                                            # two colours: not one clear colour


@pytest.mark.parametrize("det", ["smartphone", "car", "sofa", "lamp", "television", "bed"])
def test_non_clothing_detections_have_no_compatible_products(det):
    q = T.detection_query(det.title(), det, None)
    assert q.strict_types == () and q.broad_types == () and q.notes


def test_colour_relation():
    p = lambda *c: ProductInfo("tshirt", "tops", tuple(c))           # noqa: E731
    assert T.color_relation("black", p("black")) == "same"
    assert T.color_relation("black", p("grey")) == "near"
    assert T.color_relation("black", p("red")) == "different"
    assert T.color_relation(None, p("red")) == "unknown"
    assert T.color_relation("black", p()) == "unknown"


def test_taxonomy_file_is_consistent():
    """Every product type is in type_order, every detection maps to real product types."""
    raw = json.loads(Path(M.__file__).with_name("product_taxonomy.json").read_text(encoding="utf-8"))
    assert sorted(raw["type_order"]) == sorted(raw["product_types"])
    for spec in raw["detections"].values():
        assert set(spec["strict"]) | set(spec["broad"]) <= set(raw["product_types"])
        assert not set(spec["strict"]) & set(spec["broad"])


# ---------------------------------------------------------------- a fake catalogue + encoder

class FakeEncoder:
    """Embedding = normalised (mean colour, top/bottom brightness, a few pixels). Deterministic, no model."""
    model_name, pretrained = "fake", "v1"

    def __init__(self, tag="v1"):
        self.pretrained, self.calls, self.images = tag, 0, 0

    @property
    def preprocess_fingerprint(self):
        return "fakepre"

    def meta(self):
        return {"model": self.model_name, "pretrained": self.pretrained, "preprocess": self.preprocess_fingerprint}

    def embed(self, images, batch_size=64):
        self.calls += 1
        self.images += len(images)
        out = []
        for im in images:
            v = np.asarray(im.convert("RGB").resize((4, 4)), np.float32).ravel() / 255.0 - 0.5
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(out).astype(np.float32)


COL = {"black": (10, 10, 10), "white": (240, 240, 240), "red": (200, 30, 30), "blue": (30, 40, 200),
       "green": (30, 160, 60), "grey": (120, 120, 120)}


def picture(color, variant=0):
    a = np.zeros((48, 48, 3), np.uint8)
    a[:] = COL[color]
    a[: 8 + variant * 6, :] = (128, 128, 128)          # a little structure so products differ
    return Image.fromarray(a)


def make_brand(root: Path, products, slug="test_brand", layout="brand"):
    """products: (product_id, name, product_type, [(colour, variant), …])"""
    bd = root / slug
    img = bd / "images" / slug if layout == "brand" else bd / "images"
    recs = []
    for pid, name, ptype, pics in products:
        handle = f"h-{pid}"
        (img / handle).mkdir(parents=True, exist_ok=True)
        for i, (c, v) in enumerate(pics):
            picture(c, v).save(img / handle / f"{handle}-{i}.jpg", quality=95)
        recs.append({"product_id": pid, "handle": handle, "name": name, "brand": "Test Brand", "product_type": ptype,
                     "price": 100.0, "currency": "EGP", "url": f"https://example.com/{handle}",
                     "image_urls": [f"https://cdn.example.com/{handle}-{i}.jpg" for i in range(len(pics))]})
    bd.mkdir(parents=True, exist_ok=True)
    (bd / "catalogue.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return bd


CATALOGUE = [
    ("p1", "Black Leather Jacket", "Leather Jacket", [("black", 0), ("black", 1)]),
    ("p2", "Red Corduroy Jacket", "Curduroy Jackets", [("red", 0)]),
    ("p3", "Blue Fleece Jacket", "Fleece Jacket", [("blue", 0)]),
    ("p4", "Black Wide Leg Pants", "Pants", [("black", 2)]),            # same colour as the query, wrong type
    ("p5", "Black Basic Oversized Hoodie", "Plain Hoodies", [("black", 0)]),
    ("p6", "Dusty Pink Flowy Wrap", "Flowy Wraps", [("black", 0)]),     # unknown type, identical picture
    ("p7", "Grey Basic Tee", "Basic Regular Tee", [("grey", 0)]),
]


@pytest.fixture
def catalogue(tmp_path):
    root = tmp_path / "catalogues"
    bd = make_brand(root, CATALOGUE)
    enc = FakeEncoder()
    idx = M.CatalogueIndex("test_brand", bd)
    assert idx.build_or_update(encoder=enc) == 8
    mb = M.MultiBrandCatalogue([root])
    mb.discover_and_load()
    return mb, enc, root, bd


def ids(res):
    return [m.product_id for m in res["candidates"]]


# ---------------------------------------------------------------- filtering and ranking

def test_filtering_happens_before_similarity(catalogue):
    mb, enc, *_ = catalogue
    vec = enc.embed([picture("black", 0)])[0]              # identical to p1, p5 and p6 pictures
    res = mb.rank(vec, T.detection_query("Black jacket", "jacket", "black"), 3)
    assert ids(res) == ["p1", "p3", "p2"] or ids(res)[0] == "p1"
    got = {m.product_id: m for m in res["candidates"]}
    assert "p4" not in got and "p6" not in got and "p7" not in got   # pants / unknown / tee never offered
    assert all(m.normalized_type == "jacket" and m.match_tier == "strict" for m in res["candidates"])
    assert not res["fallback"] and res["pool"]["strict"] == 3
    unfiltered = mb.rank(vec, None, 7)
    assert {"p5", "p6"} <= set(ids(unfiltered))            # without the taxonomy, lookalikes of other types win


def test_best_image_per_product_and_scores(catalogue):
    mb, enc, *_ = catalogue
    vec = enc.embed([picture("black", 1)])[0]
    res = mb.rank(vec, T.detection_query("Jacket", "jacket"), 3)
    top = res["candidates"][0]
    assert top.product_id == "p1" and top.best_image_rel.endswith("h-p1-1.jpg")   # the matching image of two
    assert top.similarity == pytest.approx(1.0, abs=1e-5)
    assert len(set(ids(res))) == len(ids(res))                         # one entry per product
    sims = [m.rank_score for m in res["candidates"]]
    assert sims == sorted(sims, reverse=True)


def test_colour_reorders_but_never_excludes(catalogue):
    mb, enc, *_ = catalogue
    vec = enc.embed([picture("grey", 0)])[0]
    plain = mb.rank(vec, T.detection_query("Jacket", "jacket", None), 3)
    red = mb.rank(vec, T.detection_query("Red jacket", "jacket", "red"), 3)
    assert set(ids(plain)) == set(ids(red)) == {"p1", "p2", "p3"}      # nothing removed by colour
    assert all(m.color_relation == "unknown" for m in plain["candidates"])
    rel = {m.product_id: m.color_relation for m in red["candidates"]}
    assert rel == {"p1": "different", "p2": "same", "p3": "different"}
    for m in red["candidates"]:                                        # similarity is untouched; only the rank score moves
        assert m.rank_score == pytest.approx(m.similarity + (0.04 if m.color_relation == "same" else -0.04))
    strong = mb.rank(vec, T.detection_query("Red jacket", "jacket", "red"), 3, color_weights={"same": 1.0, "different": -1.0})
    assert ids(strong)[0] == "p2" and set(ids(strong)) == {"p1", "p2", "p3"}
    off = mb.rank(vec, T.detection_query("Red jacket", "jacket", "red"), 3, color_weights={"same": 0.0, "different": 0.0})
    assert ids(off) == ids(plain)                                      # weights are configurable


def test_fallback_to_broader_categories_is_marked_and_logged(catalogue, caplog):
    mb, enc, *_ = catalogue
    vec = enc.embed([picture("grey", 0)])[0]
    caplog.set_level(logging.INFO, logger="sceneseen.commercial.matching")
    res = mb.rank(vec, T.detection_query("Grey t-shirt", "tshirt", "grey"), 3)
    assert ids(res)[0] == "p7" and res["candidates"][0].match_tier == "strict"
    assert res["fallback"] and [m.match_tier for m in res["candidates"]][1:] == ["broad"] * (len(res["candidates"]) - 1)
    assert "p5" in ids(res)                                            # hoodie: a broader top
    assert not {"p4", "p6"} & set(ids(res))                            # never pants or unknown types
    assert "fallback" in caplog.text and "broader" in res["note"]


def test_never_fills_slots_with_unrelated_products(catalogue):
    mb, enc, *_ = catalogue
    vec = enc.embed([picture("black", 0)])[0]
    res = mb.rank(vec, T.detection_query("Smartphone", "smartphone"), 3)
    assert res["candidates"] == [] and "No product type" in res["note"]
    res = mb.rank(vec, T.detection_query("Black sneakers", "sneakers", "black"), 3)   # compatible types exist in the taxonomy,
    assert res["candidates"] == [] and "no products of a compatible type" in res["note"]   # not in this catalogue
    res = mb.rank(vec, T.detection_query("Dress", "dress"), 3)
    assert res["candidates"] == []


# ---------------------------------------------------------------- the index on disk

def test_embeddings_map_to_the_right_records(catalogue):
    mb, enc, root, bd = catalogue
    idx = mb.indices["test_brand"]
    recs = {json.loads(line)["product_id"]: json.loads(line) for line in (bd / "catalogue.jsonl").read_text().splitlines()}
    assert len(idx.items) == len(idx.embeddings) == 8
    for i, it in enumerate(idx.items):
        with Image.open(bd / it.image_path) as im:
            fresh = enc.embed([im])[0]
        assert float(fresh @ idx.embeddings[i]) == pytest.approx(1.0, abs=1e-5)       # row i is image i
        assert recs[it.product_id]["name"] == it.name and it.brand == "Test Brand"
        assert it.url == recs[it.product_id]["url"] and it.image_url.endswith(Path(it.image_filename).name)
        assert it.image_rel.startswith("test_brand/") and not Path(it.image_path).is_absolute()
        assert it.image_size and it.image_mtime_ns
    meta = json.loads((bd / "embeddings_meta.json").read_text())
    assert meta["model"] == "fake" and meta["dim"] == 48 and meta["images"] == 8 and meta["products"] == 7


def test_incremental_update_new_changed_removed(catalogue):
    mb, enc, root, bd = catalogue
    idx = M.CatalogueIndex("test_brand", bd)
    assert idx.build_or_update(encoder=enc) == 0                       # nothing changed: nothing embedded
    img_dir = bd / "images" / "test_brand"
    picture("red", 3).save(img_dir / "h-p1" / "h-p1-0.jpg", quality=95)          # changed
    picture("white", 0).save(img_dir / "h-p3" / "h-p3-9.jpg", quality=95)        # new
    (img_dir / "h-p7" / "h-p7-0.jpg").unlink()                                    # removed
    os.utime(img_dir / "h-p1" / "h-p1-0.jpg", ns=(1, 1))
    assert idx.staleness() == {"new": 1, "changed": 1, "removed": 1, "on_disk": 8}
    st = M.MultiBrandCatalogue([root]).status("fake", "v1")
    assert st["status"] == "ready" and any("new" in w and "--build-index" in w for w in st["warnings"])
    before = enc.images
    assert idx.build_or_update(encoder=enc) == 2 and enc.images - before == 2
    assert len(idx.items) == 8 and "p7" not in {it.product_id for it in idx.items}
    assert idx.staleness() == {"new": 0, "changed": 0, "removed": 0, "on_disk": 8}


def test_model_change_requires_regeneration(catalogue):
    mb, enc, root, bd = catalogue
    st = M.MultiBrandCatalogue([root]).status("ViT-B-32", "laion2b_s34b_b79k")
    assert st["status"] == "incompatible" and "fake/v1" in st["message"] and "--build-index" in st["message"]
    st = M.MultiBrandCatalogue([root]).status("fake", "v1", preprocess="other")
    assert st["status"] == "incompatible" and "preprocessing" in st["message"]
    other = FakeEncoder("v2")
    assert M.CatalogueIndex("test_brand", bd).build_or_update(encoder=other) == 8   # different model: re-embed all
    assert M.MultiBrandCatalogue([root]).status("fake", "v2")["status"] == "ready"


def test_legacy_index_without_version_info_is_adopted_not_reembedded(catalogue):
    mb, enc, root, bd = catalogue
    (bd / "embeddings_meta.json").unlink()
    man = json.loads((bd / "embeddings_manifest.json").read_text())
    for m in man:                                       # old manifests had no file stats or brand_slug
        m.pop("image_size"), m.pop("image_mtime_ns"), m.pop("brand_slug")
    (bd / "embeddings_manifest.json").write_text(json.dumps(man))
    idx = M.CatalogueIndex("test_brand", bd)
    idx.load()
    assert idx.compatibility("fake", "v1") == (False, idx.compatibility("fake", "v1")[1])   # unknown model ≠ default
    ok, why = idx.compatibility(M.DEFAULT_MODEL, M.DEFAULT_PRETRAINED)
    assert not ok                                        # fake 48-d vectors are not 512-d CLIP vectors


def test_inconsistent_or_missing_files(tmp_path, catalogue):
    mb, enc, root, bd = catalogue
    emb = np.load(bd / "embeddings.npz")["embeddings"][:5]
    np.savez_compressed(bd / "embeddings.npz", embeddings=emb)
    st = M.MultiBrandCatalogue([root]).status("fake", "v1")
    assert st["status"] == "incompatible" and "no longer belong together" in st["message"]
    (bd / "embeddings.npz").write_bytes(b"corrupt")
    assert M.MultiBrandCatalogue([root]).status("fake", "v1")["status"] == "incompatible"


def test_missing_dataset_and_not_built(tmp_path, monkeypatch):
    st = M.MultiBrandCatalogue([tmp_path / "nowhere"]).status()
    assert st["status"] == "missing_dataset" and "SCENESEEN_CATALOGUE_DIR" in st["message"]
    make_brand(tmp_path / "cat", CATALOGUE[:2])
    st = M.MultiBrandCatalogue([tmp_path / "cat"]).status()
    assert st["status"] == "not_built" and "python -m sceneseen match-catalogue --build-index" in st["message"]
    res = M.match_video_candidates([{"key": "k", "best": {"shot_id": 0, "box": [0, 0, 1, 1]}}], tmp_path,
                                   M.MultiBrandCatalogue([tmp_path / "cat"]))
    assert res["status"] == "not_built" and res["matches"] == {}      # actionable status, no crash, no silent skip


def test_catalogue_dir_from_environment(tmp_path, monkeypatch):
    from sceneseen.config import CatalogueConfig, catalogue_dirs

    monkeypatch.setenv("SCENESEEN_CATALOGUE_DIR", os.pathsep.join([str(tmp_path / "a"), "relative/dir"]))
    dirs = catalogue_dirs(CatalogueConfig())
    assert dirs[0] == tmp_path / "a" and dirs[1].is_absolute() and dirs[1].name == "dir"
    monkeypatch.delenv("SCENESEEN_CATALOGUE_DIR")
    assert [d.name for d in catalogue_dirs(CatalogueConfig())] == ["output", "catalogues"]


def test_flat_image_layout_is_supported(tmp_path):
    bd = make_brand(tmp_path, CATALOGUE[:2], slug="flat", layout="flat")
    idx = M.CatalogueIndex("flat", bd)
    assert idx.build_or_update(encoder=FakeEncoder()) == 3
    assert all((bd / it.image_path).exists() for it in idx.items)


# ---------------------------------------------------------------- whole-video matching + cache

def _video(tmp_path, color="black"):
    from sceneseen.commercial.frames import frame_name

    frames = tmp_path / "frames"
    frames.mkdir()
    picture(color, 0).resize((200, 200)).save(frames / frame_name(3, 0.5), quality=95)
    cands = [{"key": "k1", "label": f"{color.title()} jacket", "type_id": "jacket", "color": color,
              "best": {"shot_id": 3, "position": 0.5, "box": [0, 0, 1, 1]}},
             {"key": "k2", "label": "Lamp", "type_id": "lamp", "color": None,
              "best": {"shot_id": 3, "position": 0.5, "box": [0.1, 0.1, 0.9, 0.9]}},
             {"key": "k3", "label": "Watch", "type_id": "watch", "best": {"shot_id": 99, "position": 0.5, "box": [0, 0, 1, 1]}}]
    return frames, cands


def test_match_video_and_cache_invalidation(tmp_path, catalogue, monkeypatch):
    from sceneseen.config import CatalogueConfig

    mb, enc, root, bd = catalogue
    frames, cands = _video(tmp_path)
    cfg = CatalogueConfig(model="fake", pretrained="v1")
    cache = tmp_path / "vidcache"
    res = M.get_or_compute_matches(cache, cands, frames, mb, encoder=enc, cfg=cfg)
    assert res["status"] == "ready" and not res["cached"]
    k1 = res["matches"]["k1"]
    assert k1["candidates"][0]["product_id"] == "p1" and k1["query"]["color"] == "black"
    assert all(c["normalized_type"] == "jacket" for c in k1["candidates"])
    assert res["matches"]["k2"]["candidates"] == [] and "k3" not in res["matches"]    # missing frame: skipped, reported none
    calls = enc.calls
    again = M.get_or_compute_matches(cache, cands[:2], frames, mb, encoder=enc, cfg=cfg)
    assert again["cached"] and enc.calls == calls                                     # no model call
    changed = CatalogueConfig(model="fake", pretrained="v1", top_k=2)
    res2 = M.get_or_compute_matches(cache, cands[:2], frames, mb, encoder=enc, cfg=changed)
    assert not res2["cached"] and len(res2["matches"]["k1"]["candidates"]) == 2       # settings change → recomputed
    stored = json.loads((cache / "commercial" / "matches.json").read_text())
    assert stored["format"] == M.INDEX_FORMAT and stored["fingerprint"]
    (cache / "commercial" / "matches.json").write_text(json.dumps({"k1": [{"name": "old format"}]}))
    res3 = M.get_or_compute_matches(cache, cands[:2], frames, mb, allow_compute=False, encoder=enc, cfg=cfg)
    assert res3["status"] == "not_computed" and res3["matches"] == {}                  # stale old format ignored


def test_match_video_with_incompatible_index_says_what_to_do(tmp_path, catalogue):
    from sceneseen.config import CatalogueConfig

    mb, enc, root, bd = catalogue
    frames, cands = _video(tmp_path)
    res = M.get_or_compute_matches(tmp_path / "c", cands, frames, M.MultiBrandCatalogue([root]),
                                   encoder=enc, cfg=CatalogueConfig())     # configured CLIP, index made with "fake"
    assert res["status"] == "incompatible" and res["matches"] == {} and "--build-index" in res["message"]
