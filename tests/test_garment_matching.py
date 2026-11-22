"""Garment-level catalogue crops and the ranking that uses them (Phase 2B). No model is loaded:
garment boxes are written to garment_boxes.json as OWLv2 would, embeddings come from FakeEncoder."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sceneseen.commercial import garment_crop as G
from sceneseen.commercial import matching as M
from sceneseen.commercial.product_taxonomy import DetectionQuery, load_taxonomy
from sceneseen.config import CatalogueConfig
from tests.test_product_taxonomy import COL, FakeEncoder

CFG = CatalogueConfig(model="fake", pretrained="v1")


# ---------------------------------------------------------------- selection rules

def test_every_product_type_has_crop_prompts():
    tax = load_taxonomy()
    for t in tax.types:
        parts = G.crop_spec(tax, t)
        assert parts and all(parts), t
    assert G.crop_spec(tax, None) is None
    prompts = G.all_prompts(tax)
    assert len(prompts) == len(set(prompts)) and "jeans" in prompts and "pajama pants" in prompts


def test_select_uses_only_the_products_own_prompts():
    dets = [{"prompt": "t-shirt", "score": 0.9, "box": [0.2, 0.1, 0.8, 0.5]},
            {"prompt": "pants", "score": 0.4, "box": [0.25, 0.5, 0.75, 1.0]},
            {"prompt": "pants", "score": 0.3, "box": [0.0, 0.6, 0.2, 1.0]}]
    c = G.select_crop(dets, [["trousers", "pants"]], 0.15)
    assert c.method == "detector" and c.prompt == "pants" and c.box == [0.25, 0.5, 0.75, 1.0]


def test_no_confident_box_keeps_the_full_photo_with_reason():
    dets = [{"prompt": "cardigan", "score": 0.12, "box": [0.2, 0.2, 0.8, 0.8]}]
    c = G.select_crop(dets, [["cardigan", "sweater"]], 0.15)
    assert c.method == "full" and c.box == [0, 0, 1, 1] and "0.12" in c.reason
    assert G.select_crop(dets, None, 0.15).method == "full"


def test_sets_union_top_and_bottom_and_need_the_top():
    parts = [["shirt", "top"], ["pants", "shorts"]]
    dets = [{"prompt": "top", "score": 0.5, "box": [0.3, 0.1, 0.7, 0.5]},
            {"prompt": "shorts", "score": 0.4, "box": [0.25, 0.45, 0.75, 0.7]}]
    c = G.select_crop(dets, parts, 0.15)
    assert c.method == "union" and c.box == [0.25, 0.1, 0.75, 0.7] and c.score == 0.4
    only_top = G.select_crop(dets[:1], parts, 0.15)
    assert only_top.method == "detector" and only_top.box == [0.3, 0.1, 0.7, 0.5]
    assert G.select_crop(dets[1:], parts, 0.15).method == "full"


def test_padding_and_letterbox():
    assert G.pad_box([0.4, 0.4, 0.6, 0.6], 0.1) == pytest.approx([0.38, 0.38, 0.62, 0.62])
    assert G.pad_box([0.0, 0.0, 1.0, 1.0], 0.1) == [0.0, 0.0, 1.0, 1.0]
    tall = Image.new("RGB", (40, 100), (255, 0, 0))
    sq = G.letterbox(tall)
    assert sq.size == (100, 100) and sq.getpixel((50, 50)) == (255, 0, 0) and sq.getpixel((2, 50)) != (255, 0, 0)


# ---------------------------------------------------------------- garment index

def outfit(top, bottom):
    """A 'model' photo: top half = shirt colour, bottom half = trousers colour, white background."""
    a = np.full((96, 64, 3), 250, np.uint8)
    a[10:48, 12:52] = COL[top]
    a[48:92, 16:48] = COL[bottom]
    return Image.fromarray(a)


TOP_BOX, BOTTOM_BOX = [0.19, 0.1, 0.81, 0.5], [0.25, 0.5, 0.75, 0.96]
GARMENTS = [   # product id, name, shop type, (top, bottom) of each photo, which half is the product
    ("t1", "White Basic Tee", "Basic Regular Tee", [("white", "black")], "top"),
    ("t2", "Red Basic Tee", "Basic Regular Tee", [("red", "black")], "top"),
    ("b1", "Black Wide Leg Pants", "Pants", [("white", "black"), ("grey", "black")], "bottom"),
    ("b2", "Blue Wide Leg Pants", "Pants", [("white", "blue")], "bottom"),
    ("b3", "Wide Leg Pants", "Pants", [("white", "green")], "bottom"),          # no colour word in the name
]


def make_outfits(root: Path, slug="test_brand") -> Path:
    bd = root / slug
    recs = []
    for pid, name, ptype, pics, _ in GARMENTS:
        d = bd / "images" / slug / f"h-{pid}"
        d.mkdir(parents=True)
        for i, (t, b) in enumerate(pics):
            outfit(t, b).save(d / f"h-{pid}-{i}.png")
        recs.append({"product_id": pid, "handle": f"h-{pid}", "name": name, "product_type": ptype, "price": 100.0,
                     "currency": "EGP", "url": f"https://example.com/{pid}"})
    (bd / "catalogue.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return bd


def write_boxes(bd: Path, override: dict | None = None):
    """garment_boxes.json as OWLv2 would leave it: a t-shirt box on top, a pants box below."""
    tax = load_taxonomy()
    idx = M.CatalogueIndex("test_brand", bd, "garment")
    items, _ = idx.scan()
    data = {"format": G.BOXES_FORMAT, "detector": "fake", "prompts": G.all_prompts(tax), "images": {}}
    for it in items:
        dets = [{"prompt": "t-shirt", "score": 0.6, "box": TOP_BOX}, {"prompt": "pants", "score": 0.5, "box": BOTTOM_BOX}]
        data["images"][it.image_rel] = {"size": it.image_size, "mtime_ns": it.image_mtime_ns,
                                        "dets": (override or {}).get(it.product_id, dets)}
    (bd / "garment_boxes.json").write_text(json.dumps(data))


@pytest.fixture
def garments(tmp_path):
    root = tmp_path / "catalogues"
    bd = make_outfits(root)
    write_boxes(bd)
    enc = FakeEncoder()
    assert M.CatalogueIndex("test_brand", bd, "full").build_or_update(encoder=enc) == 6
    g = M.CatalogueIndex("test_brand", bd, "garment")
    assert g.build_or_update(encoder=enc, cfg=CFG) == 6
    return root, bd, enc, g


def test_garment_index_records_crops_and_is_kept_apart(garments):
    root, bd, enc, g = garments
    assert (bd / "embeddings_garment.npz").exists() and (bd / "embeddings.npz").exists()
    assert g.meta["image_mode"] == "garment" and g.meta["crop"]["letterbox"] is False
    assert json.loads((bd / "embeddings_meta.json").read_text())["image_mode"] == "full"
    by = {it.product_id: it for it in g.items}
    assert by["b1"].crop_method == "detector" and by["b1"].crop_prompt == "pants"
    assert by["b1"].crop_box == [round(v, 4) for v in G.pad_box(BOTTOM_BOX, CFG.crop_pad)]
    assert by["t1"].crop_prompt == "t-shirt"
    crop = Image.open(bd / by["b1"].crop_path)
    assert crop.height > crop.width                     # the trousers, not the whole photo
    assert by["b1"].crop_color == "black" and by["t1"].crop_color == "white" and by["b3"].crop_color == "green"
    for it in g.items:                                  # every row maps to its product, original and crop
        assert it.image_rel.startswith("test_brand/h-" + it.product_id) and (bd / it.crop_path).exists()
    # a full index read as garment (or the reverse) is refused
    wrong = M.CatalogueIndex("test_brand", bd, "garment")
    wrong.embeddings_path, wrong.manifest_path, wrong.meta_path = (bd / "embeddings.npz", bd / "embeddings_manifest.json",
                                                                   bd / "embeddings_meta.json")
    wrong.load()
    ok, why = wrong.compatibility("fake", "v1")
    assert not ok and "full embeddings, not garment" in why


def test_garment_rebuild_only_reembeds_changed_crops(garments):
    root, bd, enc, g = garments
    assert M.CatalogueIndex("test_brand", bd, "garment").build_or_update(encoder=enc, cfg=CFG) == 0
    write_boxes(bd, {"b2": [{"prompt": "pants", "score": 0.5, "box": [0.3, 0.55, 0.7, 0.95]}]})
    assert M.CatalogueIndex("test_brand", bd, "garment").build_or_update(encoder=enc, cfg=CFG) == 1
    changed = CatalogueConfig(model="fake", pretrained="v1", crop_pad=0.2)        # padding is part of the version
    idx = M.CatalogueIndex("test_brand", bd, "garment")
    idx.load()
    ok, why = idx.compatibility("fake", "v1", "fakepre", M.crop_settings(changed))
    assert not ok and "pad" in why
    assert idx.build_or_update(encoder=enc, cfg=changed) == 6


def test_unbuilt_garment_index_falls_back_to_full_photos(tmp_path):
    root = tmp_path / "catalogues"
    bd = make_outfits(root)
    M.CatalogueIndex("test_brand", bd, "full").build_or_update(encoder=FakeEncoder())
    mb = M.MultiBrandCatalogue([root], "garment")
    st = mb.status("fake", "v1", check_files=False)
    assert st["status"] == "ready" and mb.indices["test_brand"].mode == "full"
    assert any("garment-level embeddings not built yet" in w for w in st["warnings"])


def test_garment_crops_fix_outfit_confusion(garments):
    """A detected pair of black trousers: the whole photo of the WHITE tee (worn with black
    trousers) looks like it; the garment crop of the tee does not, and pants are compared with pants."""
    root, bd, enc, g = garments
    query = Image.fromarray(np.full((80, 40, 3), COL["black"], np.uint8))
    q = DetectionQuery("Black trousers", "trousers", ("trousers", "jeans"), (), "black")
    mb = M.MultiBrandCatalogue([root], "garment")
    mb.discover_and_load()
    vec = enc.embed([query])[0]
    res = mb.rank({("garment", False): vec}, q, 3, color_confidence=1.0)
    assert [m.product_id for m in res["candidates"]][0] == "b1"
    assert res["candidates"][0].best_crop_rel.startswith("crops/") and res["candidates"][0].relative_score == 0
    assert all(m.relative_score <= 0 for m in res["candidates"])


def test_colour_weight_follows_detection_confidence_and_source(garments):
    root, bd, enc, g = garments
    mb = M.MultiBrandCatalogue([root], "garment")
    mb.discover_and_load()
    vec = np.zeros(g.embeddings.shape[1], np.float32)            # all similarities 0: only colour decides
    q = DetectionQuery("Green trousers", "trousers", ("trousers",), (), "green")
    sure = {m.product_id: m for m in mb.rank(vec, q, 3, color_confidence=1.0)["candidates"]}
    assert sure["b3"].color_source == "image" and sure["b3"].color_relation == "same"
    assert sure["b3"].rank_score == pytest.approx(0.02)          # crop colour: half the name-colour weight
    assert sure["b1"].color_source == "name" and sure["b1"].rank_score == pytest.approx(-0.04)
    unsure = {m.product_id: m for m in mb.rank(vec, q, 3, color_confidence=0.25)["candidates"]}
    assert unsure["b1"].rank_score == pytest.approx(-0.01)       # an unclear colour barely moves anything
    none = {m.product_id: m for m in mb.rank(vec, q, 3, color_confidence=0.25, image_colors=False)["candidates"]}
    assert none["b3"].color_relation == "unknown"


def test_top2_aggregation(garments):
    root, bd, enc, g = garments
    mb = M.MultiBrandCatalogue([root], "garment")
    mb.discover_and_load()
    q = DetectionQuery("trousers", "trousers", ("trousers",), (), None)
    vec = g.embeddings[[i for i, it in enumerate(g.items) if it.product_id == "b1"][0]]
    mx = {m.product_id: m for m in mb.rank(vec, q, 3, aggregate="max")["candidates"]}
    t2 = {m.product_id: m for m in mb.rank(vec, q, 3, aggregate="top2")["candidates"]}
    assert mx["b1"].rank_score == pytest.approx(1.0) and t2["b1"].rank_score < 1.0
    assert t2["b2"].rank_score == pytest.approx(mx["b2"].rank_score)    # one photo: nothing to average


def test_colour_confidence():
    solid = Image.new("RGB", (40, 40), COL["black"])
    assert M.color_confidence(solid, "black") == 1.0
    assert M.color_confidence(solid, None) == 0.0 and M.color_confidence(solid, "red") == 0.0
    a = np.full((40, 40, 3), COL["red"], np.uint8)
    a[:, 20:] = COL["blue"]
    assert M.color_confidence(Image.fromarray(a), "red") < 0.5


def test_match_video_with_garment_index(tmp_path, garments):
    from sceneseen.commercial.frames import frame_name

    root, bd, enc, g = garments
    frames = tmp_path / "frames"
    frames.mkdir()
    outfit("red", "black").resize((320, 480)).save(frames / frame_name(1, 0.5))
    cands = [{"key": "pants", "label": "Black trousers", "type_id": "trousers", "color": "black",
              "best": {"shot_id": 1, "position": 0.5, "box": BOTTOM_BOX}},
             {"key": "tee", "label": "Red t-shirt", "type_id": "tshirt", "color": "red",
              "best": {"shot_id": 1, "position": 0.5, "box": TOP_BOX}}]
    mb = M.MultiBrandCatalogue([root], "garment")
    res = M.match_video_candidates(cands, frames, mb, encoder=enc, cfg=CFG)
    assert res["status"] == "ready" and res["image_mode"] == "garment"
    assert res["matches"]["pants"]["candidates"][0]["product_id"] == "b1"
    assert res["matches"]["tee"]["candidates"][0]["product_id"] == "t2"
    assert 0 < res["matches"]["pants"]["query"]["color_confidence"] <= 1


def test_layering_and_tiny_boxes():
    tee = {"prompt": "shirt", "score": 0.32, "box": [0.45, 0.2, 0.6, 0.5]}           # the tee under an open shirt
    open_shirt = {"prompt": "shirt jacket", "score": 0.6, "box": [0.3, 0.15, 0.75, 0.55]}
    c = G.select_crop([tee, open_shirt], [["shirt"]], 0.15, outer=["shirt jacket"])
    assert c.box == open_shirt["box"] and c.prompt == "shirt jacket"
    assert G.select_crop([tee, open_shirt], [["shirt"]], 0.15).box == tee["box"]     # without the rule: the tee
    beside = {"prompt": "shirt jacket", "score": 0.6, "box": [0.0, 0.1, 0.3, 0.5]}   # another person's jacket
    assert G.select_crop([tee, beside], [["shirt"]], 0.15, outer=["shirt jacket"]).box == tee["box"]
    collar = {"prompt": "shirt", "score": 0.33, "box": [0.6, 0.21, 0.67, 0.25]}
    polo = {"prompt": "polo shirt", "score": 0.27, "box": [0.47, 0.19, 0.87, 0.53]}
    assert G.select_crop([collar, polo], [["polo shirt", "shirt"]], 0.15, min_area=0.02).box == polo["box"]
    tax = load_taxonomy()
    assert G.crop_options(tax, "shirt")["outer"] == ["shirt jacket"] and G.crop_options(tax, "cap")["min_area"] == 0
