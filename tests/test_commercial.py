"""Phase 2A (commercial scene understanding) with a fake detector: no model download needed."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sceneseen.commercial import analysis as A
from sceneseen.commercial import taxonomy as T
from sceneseen.commercial.attributes import dominant_color
from sceneseen.commercial.dedup import frame_nms, max_instances, merge_shot_frames, scene_groups
from sceneseen.commercial.detector import DetectorUnavailable, get_detector
from sceneseen.commercial.frames import frame_name, frame_time
from sceneseen.commercial.models import Detection, iou
from sceneseen.commercial.relevance import commercial_relevance, size_score
from sceneseen.commercial.review import ReviewStore, attach, summarize
from sceneseen.commercial.scene_context import classify_scenes
from sceneseen.config import CommercialConfig
from sceneseen.shots import Shot

CFG = CommercialConfig()
PROMPTS = [p for p, _ in T.detector_queries()]


def q(prompt: str) -> int:
    return PROMPTS.index(prompt)


class FakeDetector:
    """Returns scripted detections per frame colour (the red channel of pixel 0,0 = shot id)."""

    def __init__(self, script):
        self.script, self.calls, self.images = script, 0, 0

    def detect(self, images, prompts):
        assert prompts == PROMPTS
        self.calls += 1
        self.images += len(images)
        return [list(self.script.get(int(im[0, 0, 0]), [])) for im in images]

    def describe(self):
        return {"name": "fake", "version": "test", "device": "cpu", "load_seconds": 0.0}


def make_video(tmp_path, n_shots=6, groups=((0, 2, 4), (1, 3, 5)), scenes=None):
    """A fake analysed video: shots, scenes, unique-shot groups and pre-extracted frames."""
    shots = [Shot(i, i * 50, (i + 1) * 50, i * 2.0, (i + 1) * 2.0) for i in range(n_shots)]
    scenes = scenes or [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": n_shots * 2.0, "shot_start": 0,
                         "shot_end": n_shots - 1}]
    uniq = []
    for gi, g in enumerate(groups):
        uniq.append({"unique_shot_id": f"S01-U{gi + 1:02d}", "representative_shot_id": g[0],
                     "occurrences": [{"shot_id": s, "start": shots[s].start, "end": shots[s].end} for s in g]})
    unique = {"scenes": [{"scene_id": 1, "unique": uniq}]}
    cache = tmp_path / "vid"
    fdir = cache / "commercial" / f"frames_{CFG.frame_long_side}"
    fdir.mkdir(parents=True)
    for g in groups:   # only representatives need a frame
        Image.new("RGB", (64, 36), (g[0], 120, 200)).save(fdir / frame_name(g[0], 0.5), quality=100, subsampling=0)
    info = SimpleNamespace(path=str(tmp_path / "missing video.mp4"), fps=25.0, duration=n_shots * 2.0)
    return cache, info, shots, scenes, unique


PHONE = (q("smartphone"), 0.80, [0.40, 0.40, 0.55, 0.70])
WATCH = (q("wristwatch"), 0.62, [0.20, 0.50, 0.26, 0.58])
SNEAK = (q("sneakers"), 0.70, [0.30, 0.80, 0.50, 0.95])
SHOES = (q("shoes"), 0.55, [0.31, 0.80, 0.50, 0.96])           # same box, competing label
CHAIR_SMALL = (q("chair"), 0.75, [0.00, 0.85, 0.04, 0.93])     # tiny background chair
WEAK = (q("handbag"), 0.15, [0.6, 0.5, 0.8, 0.8])


def run(tmp_path, script, cfg=CFG, video=None, **kw):
    cache, info, shots, scenes, unique = video or make_video(tmp_path)
    det = kw.pop("detector", None) or FakeDetector(script)
    return A.analyze(cache, info, shots, scenes, unique, cfg, detector=det, **kw), det


def shown(rec, scene=0):
    return {c["type_id"]: c for c in rec["scenes"][scene]["candidates"] if c["displayed"]}


# ---------------------------------------------------------------- taxonomy

def test_taxonomy_is_consistent_and_covers_required_categories():
    T.validate()
    assert {"fashion", "accessories", "electronics", "automotive", "food_beverage", "furniture", "beauty",
            "locations", "real_estate"} <= set(T.CATEGORIES)
    for required in ("shirt", "jacket", "dress", "trousers", "sneakers", "shoes", "watch", "sunglasses", "handbag",
                     "backpack", "jewelry", "smartphone", "laptop", "tablet", "headphones", "television", "car",
                     "motorcycle", "beverage_bottle", "soft_drink", "coffee", "food_packaging", "sofa", "chair",
                     "table", "lamp", "appliance", "cosmetics", "perfume", "skincare"):
        assert required in T.OBJECTS, required
    for venue in ("restaurant", "cafe", "hotel", "gym", "retail_store", "apartment", "villa", "office", "compound"):
        assert venue in T.VENUES and T.VENUES[venue].category in ("locations", "real_estate")


def test_scene_contexts_are_not_detector_objects():
    object_prompts = {p for p in PROMPTS}
    for c in T.VENUES.values():
        assert c.id not in T.OBJECTS and not (set(c.prompts) & object_prompts)
    assert any(c.category is None for c in T.VENUES.values())     # non-commercial distractors exist


def test_no_brand_names_in_taxonomy():
    text = " ".join([o.label for o in T.OBJECTS.values()] + PROMPTS).lower()
    for brand in ("nike", "adidas", "apple", "iphone", "samsung", "rolex", "pepsi", "coca", "gucci", "bmw"):
        assert brand not in text


# ---------------------------------------------------------------- relevance

def test_relevance_orders_by_commercial_value_and_presentation():
    box = [0.4, 0.4, 0.6, 0.6]
    watch, _ = commercial_relevance(T.OBJECTS["watch"], box, 3)
    chair, _ = commercial_relevance(T.OBJECTS["chair"], box, 3)
    assert watch > chair and 0 <= chair <= 1 and 0 <= watch <= 1
    tiny_edge, f = commercial_relevance(T.OBJECTS["chair"], [0.0, 0.9, 0.03, 0.95], 1)
    assert tiny_edge < chair and tiny_edge < CFG.min_relevance      # background furniture is hidden
    assert set(f) == {"base", "size", "centrality", "persistence"}
    once, _ = commercial_relevance(T.OBJECTS["watch"], box, 1)
    assert once < watch                                              # persistence matters
    assert once >= CFG.min_relevance                                 # ... but a visible watch is never hidden


def test_size_score_shape():
    assert size_score(0) == 0 and size_score(0.0025) == pytest.approx(0.5) and size_score(0.05) == 1.0
    assert size_score(0.95) < size_score(0.3)                        # fills the frame = probably background


# ---------------------------------------------------------------- de-duplication

def D(type_id, score, box, shot=0, pos=0.5):
    return Detection(type_id, type_id, score, box, shot, pos, f"h{shot}")


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and iou([0, 0, .5, .5], [.5, .5, 1, 1]) == 0.0


def test_frame_nms_merges_duplicates_and_competing_labels():
    dets = [D("smartphone", 0.8, [0.4, 0.4, 0.6, 0.7]), D("smartphone", 0.6, [0.41, 0.4, 0.6, 0.71]),   # same phone twice
            D("sneakers", 0.7, [0.3, 0.8, 0.5, 0.95]), D("shoes", 0.5, [0.31, 0.8, 0.5, 0.96]),          # one pair, two labels
            D("smartphone", 0.5, [0.05, 0.1, 0.15, 0.3])]                                                # a second phone elsewhere
    kept = frame_nms(dets)
    assert sorted((d.type_id, d.score) for d in kept) == [("smartphone", 0.5), ("smartphone", 0.8), ("sneakers", 0.7)]


def test_merge_shot_frames_keeps_one_object_seen_in_two_frames():
    a = [D("watch", 0.5, [0.2, 0.5, 0.3, 0.6], pos=0.15)]
    b = [D("watch", 0.7, [0.22, 0.5, 0.32, 0.6], pos=0.85), D("car", 0.6, [0.5, 0.2, 0.9, 0.6], pos=0.85)]
    merged = merge_shot_frames([a, b])
    assert sorted((d.type_id, d.score) for d in merged) == [("car", 0.6), ("watch", 0.7)]


def test_scene_groups_credit_repeated_camera_setups_once_each():
    uniq = [{"unique_shot_id": "S01-U01", "representative_shot_id": 0,
             "occurrences": [{"shot_id": s, "start": s * 2.0, "end": s * 2.0 + 2} for s in (0, 2, 4)]},
            {"unique_shot_id": "S01-U02", "representative_shot_id": 1,
             "occurrences": [{"shot_id": s, "start": s * 2.0, "end": s * 2.0 + 2} for s in (1, 3)]}]
    dets = {0: [D("smartphone", 0.8, [0.4, 0.4, 0.6, 0.7], shot=0)], 1: [D("smartphone", 0.6, [0.1, 0.1, 0.2, 0.3], shot=1)]}
    g = scene_groups(uniq, dets)
    assert list(g) == ["smartphone"]                                   # ONE candidate, not five phones
    occ = g["smartphone"]["occurrences"]
    assert [o["shot_id"] for o in occ] == [0, 1, 2, 3, 4]
    assert [o["detected"] for o in occ] == [True, True, False, False, False]
    assert g["smartphone"]["unique_shots"] == ["S01-U01", "S01-U02"]
    assert max_instances(g["smartphone"]["detections"]) == 1


# ---------------------------------------------------------------- analysis: unique shots, candidates

def test_repeated_unique_shots_are_analysed_once(tmp_path):
    rec, det = run(tmp_path, {0: [PHONE, WATCH], 1: [PHONE]})
    assert det.images == 2                       # 6 shots, 2 camera set-ups -> 2 inferences
    assert rec["status"] == "ready" and rec["summary"]["inference_avoided_by_unique_shots"] == 4
    c = shown(rec)
    assert set(c) == {"smartphone", "watch"}
    assert c["smartphone"]["seen_count"] == 6 and c["smartphone"]["detected_in_unique_shots"] == 2
    assert c["watch"]["seen_count"] == 3 and (c["watch"]["first_seen"], c["watch"]["last_seen"]) == (0.0, 10.0)
    assert c["smartphone"]["category"] == "electronics" and c["smartphone"]["category_name"] == "Electronics"
    for cand in c.values():
        assert {"detection_confidence", "commercial_relevance", "label", "best", "occurrences"} <= set(cand)
    json.dumps(rec)


def test_duplicate_and_competing_detections_become_one_candidate(tmp_path):
    rec, _ = run(tmp_path, {0: [SNEAK, SHOES, SNEAK], 1: []})
    c = shown(rec)
    assert list(c) == ["sneakers"] and c["sneakers"]["max_instances"] == 1


def test_low_confidence_and_low_relevance_are_hidden_not_dropped(tmp_path):
    rec, _ = run(tmp_path, {0: [PHONE, WEAK, CHAIR_SMALL], 1: []})
    cands = {c["type_id"]: c for c in rec["scenes"][0]["candidates"]}
    assert cands["smartphone"]["displayed"]
    assert not cands["handbag"]["displayed"] and cands["handbag"]["hidden_reason"] == "low detector confidence"
    assert not cands["chair"]["displayed"] and cands["chair"]["hidden_reason"] == "low commercial relevance"
    assert rec["summary"]["candidates_shown"] == 1 and rec["summary"]["candidates_hidden"] == 2
    assert cands["smartphone"]["debug"]["raw_label"] == "smartphone"          # developer details are kept


def test_no_detections(tmp_path):
    rec, det = run(tmp_path, {})
    assert rec["status"] == "ready" and det.images == 2
    assert rec["scenes"][0]["candidates"] == [] and rec["summary"]["candidates_shown"] == 0


def test_empty_scene_and_scene_without_unique_shots(tmp_path):
    cache, info, shots, _, unique = make_video(tmp_path)
    scenes = [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": 12.0, "shot_start": 0, "shot_end": 5},
              {"scene_id": 2, "start_seconds": 12.0, "end_seconds": 12.0, "shot_start": None, "shot_end": None}]
    rec, _ = run(tmp_path, {0: [PHONE]}, video=(cache, info, shots, scenes, unique))
    assert len(rec["scenes"]) == 2 and rec["scenes"][1]["candidates"] == []
    assert rec["scenes"][1]["unique_shots"] == 0


def test_never_names_a_brand_or_product(tmp_path):
    rec, _ = run(tmp_path, {0: [PHONE, SNEAK, WATCH]})
    for c in rec["scenes"][0]["candidates"]:
        assert c["label"].lower().split()[-1] in {o.label.lower().split()[-1] for o in T.OBJECTS.values()}


# ---------------------------------------------------------------- caching

def test_second_run_uses_cache_and_never_calls_the_model(tmp_path):
    video = make_video(tmp_path)
    first, d1 = run(tmp_path, {0: [PHONE], 1: [WATCH]}, video=video)
    second, d2 = run(tmp_path, {}, video=video)                    # different script: must not matter
    assert d1.images == 2 and d2.calls == 0
    assert second["summary"]["frames_from_cache"] == 2 and second["summary"]["frames_inferred_now"] == 0
    assert shown(second).keys() == shown(first).keys() == {"smartphone", "watch"}


def test_thresholds_and_relevance_changes_do_not_invalidate_detections(tmp_path):
    video = make_video(tmp_path)
    run(tmp_path, {0: [PHONE, WATCH]}, video=video)
    strict = CommercialConfig(min_confidence=0.7, min_relevance=0.6)
    assert A.detector_key(strict) == A.detector_key(CFG)
    rec, det = run(tmp_path, {}, cfg=strict, video=video)
    assert det.calls == 0 and set(shown(rec)) == {"smartphone"}     # watch (0.62) now hidden, no re-inference


def test_detector_or_prompt_change_invalidates_only_detections(tmp_path, monkeypatch):
    video = make_video(tmp_path)
    run(tmp_path, {0: [PHONE]}, video=video)
    assert A.detector_key(CommercialConfig(detector="grounding_dino")) != A.detector_key(CFG)
    before = A.detector_key(CFG)
    monkeypatch.setattr(T, "prompts_hash", lambda: "changed-prompts")
    assert A.detector_key(CFG) != before
    rec, det = run(tmp_path, {0: [WATCH]}, video=video)
    assert det.images == 2 and set(shown(rec)) == {"watch"}         # re-inferred with the new prompts
    frames = list((video[0] / "commercial" / f"frames_{CFG.frame_long_side}").glob("shot_*.jpg"))
    assert len(frames) == 2                                         # frames (and Phase 1) untouched


def test_new_unique_shot_only_costs_one_inference(tmp_path):
    cache, info, shots, scenes, unique = make_video(tmp_path)
    run(tmp_path, {0: [PHONE]}, video=(cache, info, shots, scenes, unique))
    # Phase 1 scenes were corrected: shot 5 is now its own camera set-up
    unique["scenes"][0]["unique"][1]["occurrences"] = [o for o in unique["scenes"][0]["unique"][1]["occurrences"] if o["shot_id"] != 5]
    unique["scenes"][0]["unique"].append({"unique_shot_id": "S01-U03", "representative_shot_id": 5,
                                          "occurrences": [{"shot_id": 5, "start": 10.0, "end": 12.0}]})
    fdir = cache / "commercial" / f"frames_{CFG.frame_long_side}"
    Image.new("RGB", (64, 36), (5, 120, 200)).save(fdir / frame_name(5, 0.5), quality=100, subsampling=0)
    rec, det = run(tmp_path, {5: [WATCH]}, video=(cache, info, shots, scenes, unique))
    assert det.images == 1 and set(shown(rec)) == {"smartphone", "watch"}


def test_corrupted_cache_is_rebuilt(tmp_path):
    video = make_video(tmp_path)
    run(tmp_path, {0: [PHONE]}, video=video)
    cache_file = video[0] / "commercial" / f"detections_{A.detector_key(CFG)}.json"
    cache_file.write_text("{ this is not json", encoding="utf-8")
    rec, det = run(tmp_path, {0: [PHONE]}, video=video)
    assert rec["status"] == "ready" and det.images == 2 and set(shown(rec)) == {"smartphone"}
    json.loads(cache_file.read_text(encoding="utf-8"))               # valid again


def test_cache_contains_no_absolute_paths(tmp_path):
    video = make_video(tmp_path)
    run(tmp_path, {0: [PHONE]}, video=video)
    for p in (video[0] / "commercial").rglob("*.json"):
        assert str(tmp_path) not in p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- failure / fallback

class Broken:
    def detect(self, images, prompts):
        raise DetectorUnavailable("could not download model: no network")


def test_model_failure_degrades_instead_of_raising(tmp_path):
    rec, _ = run(tmp_path, {}, detector=Broken())
    assert rec["status"] == "unavailable" and "no network" in rec["reason"]
    assert rec["scenes"][0]["candidates"] == []                       # still a well-formed result


def test_model_failure_keeps_cached_results_as_partial(tmp_path):
    cache, info, shots, scenes, unique = make_video(tmp_path)
    one = {"scenes": [{"scene_id": 1, "unique": unique["scenes"][0]["unique"][:1]}]}
    run(tmp_path, {0: [PHONE]}, video=(cache, info, shots, scenes, one))
    rec, _ = run(tmp_path, {}, video=(cache, info, shots, scenes, unique), detector=Broken())
    assert rec["status"] == "partial" and set(shown(rec)) == {"smartphone"}


def test_not_run_when_inference_is_not_allowed(tmp_path):
    rec, det = run(tmp_path, {0: [PHONE]}, allow_inference=False)
    assert rec["status"] == "not_run" and det.calls == 0 and rec["summary"]["candidates_shown"] == 0


def test_unsupported_frame_is_skipped(tmp_path):
    video = make_video(tmp_path)
    bad = video[0] / "commercial" / f"frames_{CFG.frame_long_side}" / frame_name(1, 0.5)
    bad.write_bytes(b"this is not an image")
    rec, det = run(tmp_path, {0: [PHONE]}, video=video)
    assert det.images == 1 and rec["summary"]["frames_skipped"] == 1
    assert set(shown(rec)) == {"smartphone"}


def test_missing_video_and_frames_is_reported_not_raised(tmp_path):
    cache, info, shots, scenes, unique = make_video(tmp_path)
    for f in (cache / "commercial").rglob("shot_*.jpg"):
        f.unlink()
    rec, det = run(tmp_path, {0: [PHONE]}, video=(cache, info, shots, scenes, unique))
    assert det.calls == 0 and rec["status"] == "unavailable"          # nothing could be analysed at all
    assert "not available" in rec["reason"] and rec["scenes"][0]["candidates"] == []


def test_unknown_detector_and_cpu_fallback_choice(monkeypatch):
    with pytest.raises(DetectorUnavailable):
        get_detector("does-not-exist")
    import torch

    from sceneseen.commercial.detector import pick_device

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device("auto") == "cpu" and pick_device("mps") == "mps"


def test_detector_load_failure_is_detector_unavailable(monkeypatch):
    import transformers

    from sceneseen.commercial.detector import MODELS, HFDetector

    def boom(*a, **k):
        raise OSError("We couldn't connect to huggingface.co")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", boom)
    with pytest.raises(DetectorUnavailable, match="could not load"):
        HFDetector(MODELS["owlv2"]).detect([np.zeros((8, 8, 3), np.uint8)], ["watch"])


# ---------------------------------------------------------------- attributes, frames, context

def test_dominant_color():
    img = np.zeros((40, 40, 3), np.uint8)
    img[:] = (250, 250, 250)
    assert dominant_color(img, [0, 0, 1, 1]) == "white"
    img[:] = (10, 10, 10)
    assert dominant_color(img, [0, 0, 1, 1]) == "black"
    img[:] = (200, 20, 20)
    assert dominant_color(img, [0, 0, 1, 1]) == "red"
    img[:, :20] = (20, 20, 200)
    assert dominant_color(img, [0, 0, 1, 1]) is None                  # no clear dominant colour -> say nothing
    assert dominant_color(img, [0.5, 0.5, 0.5, 0.5]) is None          # degenerate box


def test_frame_time_and_name():
    s = Shot(3, 75, 125, 3.0, 5.0)
    assert frame_time(s, 0.5, 25.0) == pytest.approx(4.0)
    assert 3.0 < frame_time(s, 0.0, 25.0) < frame_time(s, 1.0, 25.0) < 5.0
    assert frame_name(3, 0.5) == "shot_0003_p050.jpg"


def test_scene_context_reports_only_confident_venues():
    rng = np.random.default_rng(0)
    text = {}
    for c in [*T.VENUES.values(), *T.ENVIRONMENTS.values()]:
        for p in c.prompts:
            v = rng.normal(size=32)
            text[p] = v / np.linalg.norm(v)
    rest, off = text[T.VENUES["restaurant"].prompts[0]], text[T.VENUES["office"].prompts[0]]
    indoor = text[T.ENVIRONMENTS["indoor"].prompts[0]]
    clear = rest + 0.6 * indoor
    mixed = rest + off
    clip = np.stack([np.tile(clear, (3, 1)), np.tile(clear, (3, 1)), np.tile(mixed, (3, 1))]).astype(np.float32)
    shots = [Shot(i, i * 50, i * 50 + 50, i * 2.0, i * 2.0 + 2) for i in range(3)]
    scenes = [{"scene_id": 1, "shot_start": 0, "shot_end": 1}, {"scene_id": 2, "shot_start": 2, "shot_end": 2},
              {"scene_id": 3, "shot_start": None, "shot_end": None}]
    out = classify_scenes(clip, shots, scenes, text, 0.45, 0.10)
    assert out[0]["venue"]["id"] == "restaurant" and out[0]["venue"]["category"] == "locations"
    assert out[0]["environment"]["id"] == "indoor"
    assert out[1]["venue"] is None and {r["id"] for r in out[1]["ranked"][:2]} == {"restaurant", "office"}
    assert out[2] == {"venue": None, "environment": None, "ranked": []}
    assert "box" not in json.dumps(out)                                 # contexts never get bounding boxes


# ---------------------------------------------------------------- review (precision)

def test_review_store_precision_and_unicode(tmp_path):
    rec, _ = run(tmp_path, {0: [PHONE, WATCH, SNEAK], 1: []})
    name = "أنا بحب بيتكم ❤️ مشهد.mp4"
    store = ReviewStore(tmp_path / "gt", name)
    cands = {c["type_id"]: c for c in rec["scenes"][0]["candidates"]}
    model = {"name": "fake", "detector_key": "k1"}
    store.set_verdict(cands["smartphone"], "correct", "shady", model, T.TAXONOMY_VERSION)
    store.set_verdict(cands["watch"], "correct", "shady", model, T.TAXONOMY_VERSION)
    store.set_verdict(cands["sneakers"], "wrong", "shady", model, T.TAXONOMY_VERSION)
    store.add_missed(1, "handbag", "shady")
    assert store.path.name == "أنا بحب بيتكم ❤️ مشهد.json" and store.path.exists()
    s = summarize(tmp_path / "gt")
    d = s["displayed"]
    assert (d["reviewed"], d["correct"], d["wrong"]) == (3, 2, 1) and d["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert s["missed_reported"] == 1 and s["by_category"]["electronics"]["precision"] == 1.0
    # re-evaluate the same reviews at a stricter threshold: the wrong sneakers (0.70) and the watch (0.62) drop out
    assert summarize(tmp_path / "gt", min_confidence=0.75)["displayed"]["reviewed"] == 1
    with pytest.raises(ValueError):
        store.set_verdict(cands["watch"], "maybe", "shady", model, T.TAXONOMY_VERSION)
    store.set_verdict(cands["sneakers"], None, "shady", model, T.TAXONOMY_VERSION)       # clear a verdict
    assert summarize(tmp_path / "gt")["displayed"]["precision"] == 1.0
    attached = attach(rec, store.load())
    assert {c["type_id"]: c["review"] for c in attached["scenes"][0]["candidates"]}["smartphone"] == "correct"
    assert attached["scenes"][0]["missed"][0]["label"] == "handbag"


def test_structured_review_verdicts(tmp_path):
    """A detection is not just right or wrong: wrong label, not useful, unclear, duplicate, unsure."""
    rec, _ = run(tmp_path, {0: [PHONE, WATCH, SNEAK], 1: []})
    store = ReviewStore(tmp_path / "gt", "clip.mp4")
    c = {x["type_id"]: x for x in rec["scenes"][0]["candidates"]}
    model = {"name": "fake", "detector_key": "k1"}
    args = ("shady", model, T.TAXONOMY_VERSION)
    # wrong label -> the correct taxonomy type is required and stored
    with pytest.raises(ValueError):
        store.set_verdict(c["sneakers"], "wrong_label", *args)
    with pytest.raises(ValueError):
        store.set_verdict(c["sneakers"], "wrong_label", *args, corrected_type_id="not_a_type")
    with pytest.raises(ValueError):
        store.set_verdict(c["sneakers"], "wrong_label", *args, corrected_type_id="sneakers")      # same as detected
    store.set_verdict(c["sneakers"], "wrong_label", *args, corrected_type_id="shoes", note="formal shoes")
    r = store.load()["reviews"][c["sneakers"]["key"]]
    assert r["verdict"] == "wrong_label" and r["correction"] == {"type_id": "shoes", "label": "Shoes", "in_taxonomy": True}
    assert r["note"] == "formal shoes" and r["type_id"] == "sneakers" and r["detection_confidence"] == c["sneakers"]["detection_confidence"]
    # a label the taxonomy does not have yet is kept as free text (Arabic is fine)
    store.set_verdict(c["sneakers"], "wrong_label", *args, corrected_label="شبشب")
    assert store.load()["reviews"][c["sneakers"]["key"]]["correction"] == {"type_id": None, "label": "شبشب", "in_taxonomy": False}
    store.set_verdict(c["sneakers"], "wrong_label", *args, corrected_type_id="shoes")
    store.set_verdict(c["smartphone"], "not_commercial", *args)
    store.set_verdict(c["watch"], "correct", *args)
    s = summarize(tmp_path / "gt")
    d = s["displayed"]
    assert d["reviewed"] == 3 and d["correct"] == 1 and d["precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert d["detection_precision"] == pytest.approx(2 / 3, abs=1e-3)      # the phone WAS detected correctly
    assert d["by_verdict"]["wrong_label"] == 1 and d["by_verdict"]["not_commercial"] == 1
    assert s["label_corrections"] == [{"detected": "sneakers", "corrected": "shoes", "count": 1}]
    # unsure / unclear image say nothing about the detector: excluded from every rate
    store.set_verdict(c["smartphone"], "bad_image", *args)
    store.set_verdict(c["sneakers"], "unsure", *args)
    d = summarize(tmp_path / "gt")["displayed"]
    assert d["reviewed"] == 1 and d["excluded"] == 2 and d["precision"] == 1.0
    store.set_verdict(c["smartphone"], "duplicate", *args)
    d = summarize(tmp_path / "gt")["displayed"]
    assert d["precision"] == 0.5 and d["detection_precision"] == 1.0
    attached = attach(rec, store.load())
    by = {x["type_id"]: x for x in attached["scenes"][0]["candidates"]}
    assert by["smartphone"]["review"] == "duplicate" and by["sneakers"]["review_correction"] is None
    # the file is plain structured JSON, usable later for threshold / taxonomy work; nothing is retrained
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert {"verdict", "correction", "type_id", "detection_confidence", "commercial_relevance", "reviewer"} <= set(
        data["reviews"][c["watch"]["key"]])


def test_old_review_files_still_load(tmp_path):
    """Reviews written before the extra verdicts existed (correct / wrong only) keep working."""
    d = tmp_path / "gt" / "commercial_reviews"
    d.mkdir(parents=True)
    (d / "old.json").write_text(json.dumps({"video": "old.mp4", "missed": [], "reviews": {
        "k1": {"verdict": "correct", "type_id": "watch", "label": "Watch", "category": "accessories",
               "detection_confidence": 0.8, "commercial_relevance": 0.8},
        "k2": {"verdict": "wrong", "type_id": "watch", "label": "Watch", "category": "accessories",
               "detection_confidence": 0.7, "commercial_relevance": 0.8}}}), encoding="utf-8")
    s = summarize(tmp_path / "gt")
    assert s["displayed"]["precision"] == 0.5 and s["label_corrections"] == []
