"""Learning from ground truth: examples, training, artifacts, leakage guards, fallback."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sceneseen import grouping, learning as L
from sceneseen.config import GroupingConfig
from sceneseen.ground_truth import provenance_from_corrections, save_label
from tests.test_grouping import features_from_labels, make_shots

G = GroupingConfig()


def make_item(stem, labels, seed, tmp_path=None):
    """A labelled synthetic video: scene boundary wherever the first letter of the set-up changes
    (set-ups 'a1','a2' belong to scene a; 'b1','b2' to scene b ...)."""
    shots = make_shots([4.0] * len(labels))
    clip, color = features_from_labels(labels, seed=seed)
    # shots of one scene share a common component so that they are related but not identical
    rng = np.random.default_rng(seed + 1000)
    base = {}
    for i, lab in enumerate(labels):
        b = base.setdefault(lab[0], rng.normal(size=clip.shape[2]))
        clip[i] = clip[i] + 0.9 * b / np.linalg.norm(b)
        clip[i] /= np.linalg.norm(clip[i], axis=1, keepdims=True)
    gt = [shots[i].start for i in range(1, len(labels)) if labels[i][0] != labels[i - 1][0]]
    label_path = None
    if tmp_path is not None:
        label_path = save_label(tmp_path, f"{stem}.mp4", gt, annotator="test")
    return {"stem": stem, "gt": gt, "label_path": label_path,
            "info": SimpleNamespace(duration=shots[-1].end, fps=25.0),
            "stage": {"shots": shots, "features": {"clip": clip, "color": color}, "shots_rec": {}}}


SCRIPT = ["a1", "a2", "a1", "a2", "a1", "b1", "b2", "b1", "b2", "c1", "c2", "c1", "c2", "c1", "d1", "d2", "d1"]


def dataset(tmp_path=None, n=4):
    return [make_item(f"video_{k}", SCRIPT[k:] + SCRIPT[:k], seed=k, tmp_path=tmp_path) for k in range(n)]


# ---- ground truth -> examples

def test_boundary_labels_positive_and_negative():
    cuts = np.array([4.0, 8.0, 12.0, 16.0])
    y, unreachable = L.boundary_labels(cuts, [8.9, 16.0])
    assert y.tolist() == [0, 1, 0, 1] and unreachable == []     # 8.9 s label -> the 8.0 s cut (within 2 s)


def test_boundary_labels_one_to_one_and_unreachable():
    y, unreachable = L.boundary_labels(np.array([10.0, 20.0]), [10.5, 11.0, 50.0])
    assert y.sum() == 1                      # two labels near one cut: only one positive
    assert 50.0 in unreachable               # no shot cut near 50 s: cannot be learned or predicted


def test_build_examples_shapes_and_targets():
    it = make_item("v", SCRIPT, seed=0)
    ex = L.build_examples(it, G)
    n_cuts = len(SCRIPT) - 1
    assert ex["X"].shape == (n_cuts, len(L.FEATURES)) and np.isfinite(ex["X"]).all()
    assert ex["y"].shape == (n_cuts,) and ex["y"].sum() == len(it["gt"]) == 3
    assert set(np.unique(ex["y"])) == {0, 1}
    # positives sit exactly on the cuts where the scene letter changes
    assert [SCRIPT[k + 1] for k in np.where(ex["y"] == 1)[0]] == ["b1", "c1", "d1"]


def test_single_shot_video_has_no_examples():
    it = make_item("v", ["a1"], seed=0)
    ex = L.build_examples(it, G)
    assert ex["X"].shape == (0, len(L.FEATURES)) and len(ex["y"]) == 0


def test_correction_log_becomes_hard_examples():
    corr = {"predicted_boundaries": [10.0, 20.0, 30.0],
            "log": [{"op": "merge", "scene_index": 1}, {"op": "split", "at": 44.0}, {"op": "confirm"}]}
    ex = L.correction_examples(corr, cut_times=[10.0, 20.0, 30.0, 44.0])
    assert ex == [{"time": 20.0, "label": 0, "kind": "hard_negative", "cut": 1},
                  {"time": 44.0, "label": 1, "kind": "hard_positive", "cut": 3}]
    assert provenance_from_corrections(corr)["label_method"] == "edited_predictions"
    assert provenance_from_corrections({"log": [{"op": "clear"}, {"op": "split", "at": 5}]})["label_method"] == "blank"


# ---- training

def test_training_learns_and_is_deterministic():
    ex = [L.build_examples(it, G) for it in dataset()]
    m1, hp1 = L.train_model(ex)
    m2, hp2 = L.train_model(ex)
    assert hp1 == hp2 and np.array_equal(m1.coef, m2.coef) and m1.intercept == m2.intercept
    held_out = L.build_examples(make_item("new", SCRIPT[3:] + SCRIPT[:3], seed=99), G)
    pred = m1.boundaries(held_out["times"], held_out["X"], held_out["duration"])
    assert L._f1(held_out, pred) >= 0.8                       # generalises to an unseen synthetic video


def test_out_of_fold_predictions_never_use_the_scored_video(monkeypatch):
    ex = [L.build_examples(it, G) for it in dataset()]
    seen = []

    def spy(Xtr, ytr, Xte, C):
        seen.append((len(Xtr), len(Xte)))
        return np.zeros(len(Xte))

    monkeypatch.setitem(L.PROBA, "logreg", spy)
    L.out_of_fold(ex, "logreg", 1.0)
    total = sum(len(e["X"]) for e in ex)
    assert all(n_train == total - n_test for n_train, n_test in seen)   # held-out rows excluded from the fit


def test_validation_videos_are_never_fitted(monkeypatch):
    dev, val = dataset(n=3), [make_item("val_video", SCRIPT[5:] + SCRIPT[:5], seed=50)]
    fitted_sizes = []
    real = L._fit_logreg

    def spy(X, y, C):
        fitted_sizes.append(len(X))
        return real(X, y, C)

    monkeypatch.setattr(L, "_fit_logreg", spy)
    rep = L.run_experiment(dev, val, G, with_tuned=False)
    dev_rows = sum(len(L.build_examples(it, G)["X"]) for it in dev)
    assert max(fitted_sizes) <= dev_rows                    # no fit ever included the validation rows
    assert [r["video"] for r in rep["val"]["C_logreg"]] == ["val_video"]
    assert {r["video"] for r in rep["dev_loo"]["C_logreg"]} == {"video_0", "video_1", "video_2"}


def test_final_test_leakage_is_refused():
    split = {"dev": ["a", "b"], "val": ["c"], "test": ["d"]}
    L.assert_no_test_leak(["a", "b"], split)
    with pytest.raises(RuntimeError, match="leak"):
        L.assert_no_test_leak(["a", "d"], split)
    with pytest.raises(RuntimeError, match="leak"):
        L.assert_no_test_leak(["a", "c"], split)


# ---- artifact

def test_model_save_load_roundtrip_with_unicode(tmp_path):
    items = dataset(tmp_path)
    items[0]["stem"] = "أنا بحب بيتكم ❤️ مشهد"                 # Arabic + emoji video name in the metadata
    ex = [L.build_examples(it, G) for it in items]
    model, hp = L.train_model(ex)
    model.meta = L.training_metadata(items, ex, hp, G, {"accepted": False})
    path = model.save(tmp_path / "نموذج" / "boundary_logreg.json")   # Arabic directory name
    loaded = L.BoundaryModel.load(path)
    assert np.allclose(loaded.predict_proba(ex[1]["X"]), model.predict_proba(ex[1]["X"]))
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["training_videos"][0]["stem"] == "أنا بحب بيتكم ❤️ مشهد"
    for key in ("training_videos", "ground_truth_version", "features", "algorithm", "hyperparameters",
                "random_seed", "artifact_hash", "created", "threshold"):
        assert key in d
    assert all(len(v["label_sha256"]) == 64 for v in d["training_videos"])


def test_tampered_or_incompatible_model_is_rejected(tmp_path):
    ex = [L.build_examples(it, G) for it in dataset()]
    model, _ = L.train_model(ex)
    path = model.save(tmp_path / "m.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    d["threshold"] = 0.01
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="modified"):
        L.BoundaryModel.load(path)
    d = model.to_dict()
    d["features"] = d["features"][:-1]
    d.pop("artifact_hash")
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="feature set"):
        L.BoundaryModel.load(path)


def test_fallback_to_rule_when_no_or_bad_model(tmp_path):
    assert L.load_model_or_none("", tmp_path) is None                       # default: rule
    assert L.load_model_or_none("models/does_not_exist.json", tmp_path) is None
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        L.BoundaryModel.load(tmp_path / "broken.json")


def test_pipeline_uses_rule_by_default_and_model_only_when_given():
    from sceneseen.pipeline import build_result

    it = make_item("v", SCRIPT, seed=0)
    info = SimpleNamespace(path="v.mp4", video_id="x", duration=it["info"].duration, fps=25.0)
    stage = {**it["stage"], "shots_rec": {"detector": "test", "raw_shot_count": len(SCRIPT)}, "feat_key": "k"}
    rule = build_result(info, stage, G, {})
    direct = grouping.group_shots(stage["shots"], stage["features"]["clip"], stage["features"]["color"],
                                  info.duration, G)
    assert rule["boundaries"] == direct["boundaries"] and rule["debug"]["boundary_method"] == "rule"
    model, _ = L.train_model([L.build_examples(x, G) for x in dataset()])
    learned = build_result(info, stage, G, {}, model)
    assert learned["debug"]["boundary_method"].startswith("learned:")
    assert all("probability" in c and "rule_decision" in c for c in learned["debug"]["cuts"])
    assert build_result(info, stage, G, {})["boundaries"] == rule["boundaries"]   # rule path unchanged afterwards


def test_select_by_probability_respects_min_scene_length():
    times, prob = np.array([10.0, 12.0, 30.0]), np.array([0.9, 0.8, 0.7])
    assert L.select_by_probability(times, prob, 60.0, 0.5, 6.0) == [10.0, 30.0]
    assert L.select_by_probability(times, prob, 60.0, 0.95, 6.0) == []
