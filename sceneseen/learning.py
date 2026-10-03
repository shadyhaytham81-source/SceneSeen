"""Learning scene boundaries from human ground truth.

What is (and is not) trainable in SceneSeen
-------------------------------------------
* TransNetV2 (shot detection) and OpenCLIP (frame embeddings) are pretrained networks used
  FROZEN. A handful of labelled videos cannot retrain them.
* The scene-grouping layer (grouping.py) is hand-designed logic with ~11 tunable numbers
  (thresholds, window sizes, weights). It has no learned weights.
* The only genuinely trainable component is defined here: a small supervised classifier that
  looks at every shot cut and predicts "scene boundary" (1) or "same scene" (0) from signals
  SceneSeen already computes. It replaces the hand-set thresholds, not the vision models.

Pipeline:  ground truth -> one example per shot cut -> classifier -> probabilities
           -> boundaries (probability threshold + the same minimum-scene-length rule).

The classifier is OPTIONAL. With no model configured (the default) SceneSeen uses the
hand-designed rule; a model that fails to load also falls back to the rule.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import grouping
from .config import GroupingConfig
from .evaluation import evaluate, match_boundaries
from .shots import Shot

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
LABEL_TOLERANCE = 2.0  # a labelled boundary belongs to the nearest shot cut within this many seconds

FEATURES = [
    "score",            # 1 - windowed coherence (the rule's main signal)
    "depth_climb",      # TextTiling depth of the score
    "depth_window",     # prominence vs. lowest score in the neighbourhood
    "score_rel",        # score minus the median score of nearby cuts (video-independent scale)
    "score_w2",         # same score with a short context window (2 shots)
    "score_w10",        # ... and a long one (10 shots)
    "adj_clip",         # CLIP cosine similarity of the two shots at the cut
    "adj_color",        # colour-histogram similarity of the two shots at the cut
    "cross_mean",       # mean similarity between the 4 shots before and the 4 after
    "left_coherence",   # how alike the 4 shots before the cut are (is the left side one scene?)
    "right_coherence",  # same for the 4 shots after
    "log_dur_prev",     # log duration of the shot before the cut
    "log_dur_next",     # log duration of the shot after the cut
    "pace",             # cuts in the surrounding 30 s (editing pace)
    "trans_peak",       # shot detector's transition probability at the cut
    "trans_gradual",    # 1 if the transition spans several frames (dissolve / fade)
]


# ---------------------------------------------------------------- features

def _block_mean(S: np.ndarray, rows: range, cols: range, exclude_diag: bool = False) -> float:
    if len(rows) == 0 or len(cols) == 0:
        return 0.0
    sub = S[np.ix_(list(rows), list(cols))]
    if exclude_diag:
        if len(rows) < 2:
            return 1.0
        return float((sub.sum() - np.trace(sub)) / (sub.size - len(rows)))
    return float(sub.mean())


def boundary_features(shots: list[Shot], clip: np.ndarray, color: np.ndarray, g: GroupingConfig,
                      transition_probs: list[float] | None = None, fps: float = 25.0) -> np.ndarray:
    """[n_cuts, len(FEATURES)] feature matrix; cut k lies between shot k and shot k+1."""
    n = len(shots)
    if n < 2:
        return np.zeros((0, len(FEATURES)), np.float32)
    S = grouping.similarity_matrix(clip, color, g)
    cs = grouping.clip_similarity(clip)
    hs = grouping.color_similarity(color)
    score = np.array([c.score for c in grouping.cut_scores(S, shots, g)])
    s2 = np.array([c.score for c in grouping.cut_scores(S, shots, g.replace(window_shots=2, coherence_topk=1))])
    s10 = np.array([c.score for c in grouping.cut_scores(S, shots, g.replace(window_shots=10))])
    d_climb = grouping.depth_scores(score)
    d_win = grouping.window_depth_scores(score, max(1, g.window_shots // 2))
    times = np.array([s.start for s in shots[1:]])
    probs = np.asarray(transition_probs, np.float32) if transition_probs else None
    X = np.zeros((n - 1, len(FEATURES)), np.float32)
    for k in range(n - 1):
        left, right = range(max(0, k - 3), k + 1), range(k + 1, min(n, k + 5))
        lo, hi = max(0, k - 5), min(n - 1, k + 6)
        peak, gradual = 0.0, 0.0
        if probs is not None and len(probs):
            f = int(round(times[k] * fps))
            w = probs[max(0, f - 12):min(len(probs), f + 12)]
            if len(w):
                peak, gradual = float(w.max()), float((w > 0.5).sum() > 2)
        X[k] = [
            score[k], d_climb[k], d_win[k], score[k] - float(np.median(score[lo:hi])), s2[k], s10[k],
            cs[k, k + 1], hs[k, k + 1], _block_mean(S, left, right),
            _block_mean(S, left, left, exclude_diag=True), _block_mean(S, right, right, exclude_diag=True),
            np.log1p(shots[k].end - shots[k].start), np.log1p(shots[k + 1].end - shots[k + 1].start),
            float(((times > times[k] - 15) & (times < times[k] + 15)).sum()),
            peak, gradual,
        ]
    return X


def boundary_labels(cut_times: np.ndarray, gt: list[float]) -> tuple[np.ndarray, list[float]]:
    """y[k] = 1 if a labelled boundary is assigned to cut k (optimal one-to-one matching within
    LABEL_TOLERANCE). Also returns labelled boundaries with no shot cut nearby (unreachable:
    the shot detector produced no cut there, so no cut-level classifier can find them)."""
    y = np.zeros(len(cut_times), np.int64)
    matched = match_boundaries(cut_times, gt, LABEL_TOLERANCE)
    for ci, _, _ in matched:
        y[ci] = 1
    got = {gi for _, gi, _ in matched}
    return y, [float(b) for i, b in enumerate(gt) if i not in got]


def build_examples(item: dict, g: GroupingConfig) -> dict:
    """Training/evaluation examples for one labelled video (an item from benchmark.load_dataset)."""
    st, info = item["stage"], item["info"]
    shots = st["shots"]
    X = boundary_features(shots, st["features"]["clip"], st["features"]["color"], g,
                          st["shots_rec"].get("transition_probs"), info.fps)
    times = np.array([s.start for s in shots[1:]])
    y, unreachable = boundary_labels(times, item["gt"])
    return {"stem": item["stem"], "X": X, "y": y, "times": times, "duration": info.duration,
            "gt": list(item["gt"]), "unreachable": unreachable}


def correction_examples(corrections: dict, cut_times: list[float]) -> list[dict]:
    """Turn a stored correction log (corrections.json) into explicit 'hard' examples:
    every merge = the model predicted a boundary a human rejected (hard negative);
    every split = a boundary the model missed (hard positive)."""
    cuts = np.asarray(cut_times)
    out = []
    bounds = sorted(corrections.get("predicted_boundaries", []))
    for op in corrections.get("log", []):
        if op["op"] == "clear":
            bounds = []
        elif op["op"] == "merge" and 0 <= op.get("scene_index", -1) < len(bounds):
            t = bounds.pop(op["scene_index"])
            out.append({"time": t, "label": 0, "kind": "hard_negative"})
        elif op["op"] == "split":
            t = float(op["at"])
            bounds = sorted([*bounds, t])
            out.append({"time": t, "label": 1, "kind": "hard_positive"})
    for e in out:
        e["cut"] = int(np.argmin(np.abs(cuts - e["time"]))) if len(cuts) else None
    return out


# ---------------------------------------------------------------- decoding

def select_by_probability(times: np.ndarray, prob: np.ndarray, duration: float, threshold: float,
                          min_scene_seconds: float) -> list[float]:
    """Boundaries = cuts with probability >= threshold, strongest first, keeping scenes at least
    min_scene_seconds long (the same non-maximum suppression the rule uses)."""
    chosen: list[float] = []
    for k in np.argsort(-prob, kind="stable"):
        if prob[k] < threshold:
            break
        t = float(times[k])
        if min(abs(t - e) for e in [0.0, duration, *chosen]) >= min_scene_seconds:
            chosen.append(t)
    return sorted(round(t, 3) for t in chosen)


def hybrid_select(times: np.ndarray, prob: np.ndarray, rule_boundaries: list[float], duration: float,
                  veto_below: float, add_above: float, min_scene_seconds: float) -> list[float]:
    """Hybrid: keep a rule boundary unless the classifier is confident it is wrong (p < veto_below),
    and add cuts the classifier is confident about (p >= add_above)."""
    rule = np.array([any(abs(t - b) < 1e-3 for b in rule_boundaries) for t in times])
    keep = (rule & (prob >= veto_below)) | (prob >= add_above)
    return select_by_probability(times, np.where(keep, np.maximum(prob, 1e-6), 0.0), duration, 1e-9,
                                 min_scene_seconds)


# ---------------------------------------------------------------- model (numpy inference, JSON artifact)

def _artifact_hash(body: dict) -> str:
    """Version of a trained model: hash of everything except the hash itself and the creation
    time, so retraining on identical data with identical code gives the identical version."""
    core = {k: v for k, v in body.items() if k not in ("artifact_hash", "created")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


class BoundaryModel:
    """Standardise -> linear -> sigmoid. Stored as plain JSON so a trained model can be loaded
    on any machine without pickle and without scikit-learn."""

    def __init__(self, features: list[str], mean, scale, coef, intercept: float, threshold: float,
                 min_scene_seconds: float, meta: dict | None = None):
        self.features = list(features)
        self.mean, self.scale = np.asarray(mean, np.float64), np.asarray(scale, np.float64)
        self.coef, self.intercept = np.asarray(coef, np.float64), float(intercept)
        self.threshold, self.min_scene_seconds = float(threshold), float(min_scene_seconds)
        self.meta = meta or {}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def boundaries(self, times: np.ndarray, X: np.ndarray, duration: float) -> list[float]:
        return select_by_probability(times, self.predict_proba(X), duration, self.threshold, self.min_scene_seconds)

    def to_dict(self) -> dict:
        body = {
            "format_version": FORMAT_VERSION, "algorithm": "logistic_regression", "features": self.features,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
            "intercept": self.intercept, "threshold": self.threshold, "min_scene_seconds": self.min_scene_seconds,
            **self.meta,
        }
        body.pop("artifact_hash", None)
        body["artifact_hash"] = _artifact_hash(body)
        return body

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BoundaryModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported boundary model format {d.get('format_version')}")
        if d["features"] != FEATURES:
            raise ValueError("boundary model was trained with a different feature set than this code computes")
        stored = d.get("artifact_hash")
        if stored and _artifact_hash(d) != stored:
            raise ValueError("boundary model file was modified after training (hash mismatch)")
        core = ("format_version", "algorithm", "features", "mean", "scale", "coef", "intercept", "threshold",
                "min_scene_seconds", "artifact_hash")
        return cls(d["features"], d["mean"], d["scale"], d["coef"], d["intercept"], d["threshold"],
                   d["min_scene_seconds"], {k: v for k, v in d.items() if k not in core})


def load_model_or_none(path: str, root: Path) -> BoundaryModel | None:
    """Configured model, or None (-> hand-designed rule) when no model is set or it cannot be used."""
    if not path:
        return None
    p = Path(path) if Path(path).is_absolute() else root / path
    try:
        return BoundaryModel.load(p)
    except (OSError, ValueError, KeyError) as e:
        log.warning("boundary model %s not usable (%s); falling back to the rule-based grouping", p, e)
        return None


# ---------------------------------------------------------------- training

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_SCENES = [6.0, 10.0]
C_GRID = [0.03, 0.3, 3.0]
SEED = 13


def _fit_logreg(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    scale = np.where(sc.scale_ < 1e-9, 1.0, sc.scale_)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED).fit((X - sc.mean_) / scale, y)
    return sc.mean_, scale, m.coef_[0], float(m.intercept_[0])


def _proba_logreg(Xtr, ytr, Xte, C):
    mean, scale, coef, b = _fit_logreg(Xtr, ytr, C)
    return 1.0 / (1.0 + np.exp(-(((Xte - mean) / scale) @ coef + b)))


def _proba_gbdt(Xtr, ytr, Xte, C):  # C unused; fixed small, shallow model
    from sklearn.ensemble import GradientBoostingClassifier

    m = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8,
                                   random_state=SEED).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


PROBA = {"logreg": _proba_logreg, "gbdt": _proba_gbdt}


def _f1(ex: dict, bounds: list[float]) -> float:
    return evaluate(bounds, ex["gt"], ex["duration"], (2.0,))["by_tolerance"]["2"]["f1"]


def out_of_fold(examples: list[dict], algo: str, C: float) -> list[np.ndarray]:
    """Leave-one-video-out probabilities for every video in `examples`."""
    out = []
    for i, ex in enumerate(examples):
        rest = examples[:i] + examples[i + 1:]
        Xtr, ytr = np.concatenate([e["X"] for e in rest]), np.concatenate([e["y"] for e in rest])
        out.append(PROBA[algo](Xtr, ytr, ex["X"], C) if len(ex["X"]) else np.zeros(0))
    return out


def choose_hyperparameters(examples: list[dict], algo: str) -> dict:
    """Pick C / threshold / min-scene-length using ONLY the given (training) videos, scoring each
    setting on leave-one-video-out predictions so the choice is not made on memorised data."""
    best = {"score": -1.0}
    for C in (C_GRID if algo == "logreg" else [None]):
        oof = out_of_fold(examples, algo, C)
        for thr in THRESHOLDS:
            for ms in MIN_SCENES:
                s = float(np.mean([_f1(ex, select_by_probability(ex["times"], p, ex["duration"], thr, ms))
                                   for ex, p in zip(examples, oof)]))
                if s > best["score"] + 1e-9:
                    best = {"score": s, "C": C, "threshold": thr, "min_scene_seconds": ms}
    return best


def choose_hybrid(examples: list[dict], rule_bounds: list[list[float]], C: float, min_scene: float) -> dict:
    oof = out_of_fold(examples, "logreg", C)
    best = {"score": -1.0}
    for veto in [0.0, 0.05, 0.1, 0.2, 0.3]:
        for add in [0.6, 0.7, 0.8, 0.9, 1.01]:
            s = float(np.mean([_f1(ex, hybrid_select(ex["times"], p, rb, ex["duration"], veto, add, min_scene))
                               for ex, p, rb in zip(examples, oof, rule_bounds)]))
            if s > best["score"] + 1e-9:
                best = {"score": s, "veto_below": veto, "add_above": add}
    return best


def assert_no_test_leak(train_stems: list[str], split: dict) -> None:
    """Hard guard: training data must not contain validation or final-test videos."""
    from .ground_truth import nfc

    held = {nfc(s) for s in split.get("test", [])} | {nfc(s) for s in split.get("val", [])}
    leak = sorted(held & {nfc(s) for s in train_stems})
    if leak:
        raise RuntimeError(f"data leakage: training set contains held-out videos: {leak}")


def train_model(examples: list[dict], algo: str = "logreg") -> tuple[BoundaryModel, dict]:
    """Fit the final (logistic) model on all given training videos with hyperparameters chosen
    by leave-one-video-out inside the training set. Deterministic for a fixed input."""
    if algo != "logreg":
        raise ValueError("only the logistic model is saved as a deployable artifact")
    hp = choose_hyperparameters(examples, "logreg")
    X, y = np.concatenate([e["X"] for e in examples]), np.concatenate([e["y"] for e in examples])
    mean, scale, coef, b = _fit_logreg(X, y, hp["C"])
    return BoundaryModel(FEATURES, mean, scale, coef, b, hp["threshold"], hp["min_scene_seconds"]), hp


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_metadata(items: list[dict], examples: list[dict], hp: dict, g: GroupingConfig, extra: dict) -> dict:
    vids = [{"stem": it["stem"], "label_sha256": file_sha256(it["label_path"]), "cuts": int(len(ex["y"])),
             "positives": int(ex["y"].sum()), "unreachable_boundaries": len(ex["unreachable"])}
            for it, ex in zip(items, examples)]
    gt_version = hashlib.sha256("".join(sorted(v["label_sha256"] for v in vids)).encode()).hexdigest()[:16]
    import sklearn

    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "random_seed": SEED,
        "hyperparameters": {"C": hp["C"], "class_weight": "balanced", "selected_by": "leave-one-video-out on training videos"},
        "training_videos": vids, "ground_truth_version": gt_version,
        "grouping_config_for_features": asdict(g), "label_tolerance_seconds": LABEL_TOLERANCE,
        "sklearn_version": sklearn.__version__, **extra,
    }


# ---------------------------------------------------------------- experiment: rule vs tuned vs learned

METHODS = ["A_current_rule", "B_tuned_rule", "C_logreg", "C_gbdt", "D_hybrid"]


def _full_metrics(ex: dict, bounds: list[float]) -> dict:
    ev = evaluate(bounds, ex["gt"], ex["duration"], (1.0, 2.0, 3.0))
    bt = ev["by_tolerance"]
    return {"precision": bt["2"]["precision"], "recall": bt["2"]["recall"], "f1_1s": bt["1"]["f1"],
            "f1_2s": bt["2"]["f1"], "f1_3s": bt["3"]["f1"], "tp": bt["2"]["tp"], "fp": bt["2"]["fp"],
            "fn": bt["2"]["fn"], "pred_scenes": ev["n_pred_scenes"], "true_scenes": ev["n_gt_scenes"],
            "scene_count_error": ev["n_pred_scenes"] - ev["n_gt_scenes"], "coverage": ev["coverage"],
            "overflow": ev["overflow"]}


def _predict_all(train_items, train_ex, test_items, test_ex, g: GroupingConfig, with_tuned: bool = True) -> dict:
    """Fit every method on the training videos only, predict the test videos."""
    from . import benchmark as bm

    rule = lambda it, cfg_: bm.predict(it, "sceneseen", cfg_)  # noqa: E731
    out = {m: [] for m in METHODS}
    info = {}
    # B: grid-tuned rule (tuned on training videos only)
    g_tuned = bm._best(train_items, list(bm._grid(g)))[0] if with_tuned else g
    # C: classifiers with hyperparameters chosen inside the training videos
    Xtr, ytr = np.concatenate([e["X"] for e in train_ex]), np.concatenate([e["y"] for e in train_ex])
    hp_lr, hp_gb = choose_hyperparameters(train_ex, "logreg"), choose_hyperparameters(train_ex, "gbdt")
    hp_hy = choose_hybrid(train_ex, [rule(it, g) for it in train_items], hp_lr["C"], g.min_scene_seconds)
    info = {"tuned_rule": {k: v for k, v in asdict(g_tuned).items() if asdict(g)[k] != v},
            "logreg": hp_lr, "gbdt": hp_gb, "hybrid": hp_hy}
    for it, ex in zip(test_items, test_ex):
        rb = rule(it, g)
        p_lr = _proba_logreg(Xtr, ytr, ex["X"], hp_lr["C"]) if len(ex["X"]) else np.zeros(0)
        p_gb = _proba_gbdt(Xtr, ytr, ex["X"], None) if len(ex["X"]) else np.zeros(0)
        out["A_current_rule"].append(rb)
        out["B_tuned_rule"].append(rule(it, g_tuned))
        out["C_logreg"].append(select_by_probability(ex["times"], p_lr, ex["duration"], hp_lr["threshold"],
                                                     hp_lr["min_scene_seconds"]))
        out["C_gbdt"].append(select_by_probability(ex["times"], p_gb, ex["duration"], hp_gb["threshold"],
                                                   hp_gb["min_scene_seconds"]))
        out["D_hybrid"].append(hybrid_select(ex["times"], p_lr, rb, ex["duration"], hp_hy["veto_below"],
                                             hp_hy["add_above"], g.min_scene_seconds))
    return {"predictions": out, "selected": info}


def run_experiment(dev_items: list[dict], val_items: list[dict], g: GroupingConfig, with_tuned: bool = True) -> dict:
    """Compare the current rule, the tuned rule, the learned classifiers and a hybrid.

    * development: leave-one-video-out over the dev videos (every number is for a video the
      method did not see while being fitted or tuned);
    * validation: fitted/tuned on ALL dev videos, scored on the held-out validation videos.
    Final-test videos are never loaded here.
    """
    dev_ex = [build_examples(it, g) for it in dev_items]
    val_ex = [build_examples(it, g) for it in val_items]
    rep: dict = {"dev_loo": {m: [] for m in METHODS}, "val": {m: [] for m in METHODS}, "selected": {}}
    for i in range(len(dev_items)):
        tr_i, tr_e = dev_items[:i] + dev_items[i + 1:], dev_ex[:i] + dev_ex[i + 1:]
        r = _predict_all(tr_i, tr_e, [dev_items[i]], [dev_ex[i]], g, with_tuned)
        for m in METHODS:
            rep["dev_loo"][m].append({"video": dev_items[i]["stem"], **_full_metrics(dev_ex[i], r["predictions"][m][0])})
    if val_items:
        r = _predict_all(dev_items, dev_ex, val_items, val_ex, g, with_tuned)
        rep["selected"] = r["selected"]
        for m in METHODS:
            for it, ex, b in zip(val_items, val_ex, r["predictions"][m]):
                rep["val"][m].append({"video": it["stem"], **_full_metrics(ex, b)})
    # training-set fit (reported only to show the optimism of in-sample numbers)
    r = _predict_all(dev_items, dev_ex, dev_items, dev_ex, g, with_tuned)
    rep["dev_in_sample"] = {m: [{"video": it["stem"], **_full_metrics(ex, b)}
                                for it, ex, b in zip(dev_items, dev_ex, r["predictions"][m])] for m in METHODS}
    rep["examples"] = {"dev": [{"video": e["stem"], "cuts": int(len(e["y"])), "positives": int(e["y"].sum()),
                                "unreachable": e["unreachable"]} for e in dev_ex],
                       "val": [{"video": e["stem"], "cuts": int(len(e["y"])), "positives": int(e["y"].sum()),
                                "unreachable": e["unreachable"]} for e in val_ex]}

    def macro(rows, key="f1_2s"):
        return float(np.mean([r_[key] for r_ in rows])) if rows else None

    rep["summary"] = {part: {m: {k: macro(rep[part][m], k) for k in
                                 ("precision", "recall", "f1_1s", "f1_2s", "f1_3s", "coverage", "overflow")}
                             | {"mean_abs_scene_count_error": macro([{"e": abs(x["scene_count_error"])} for x in rep[part][m]], "e")}
                             for m in METHODS} for part in ("dev_in_sample", "dev_loo", "val")}
    return rep


def decide(rep: dict) -> dict:
    """A learned/tuned method replaces the current rule only if it is better on held-out data
    in BOTH protocols and does not lose on more videos than it wins."""
    base_dev, base_val = rep["summary"]["dev_loo"]["A_current_rule"]["f1_2s"], rep["summary"]["val"]["A_current_rule"]["f1_2s"]
    verdicts = {}
    for m in METHODS[1:]:
        dv, vv = rep["summary"]["dev_loo"][m]["f1_2s"], rep["summary"]["val"][m]["f1_2s"]
        rows = rep["dev_loo"][m] + rep["val"][m]
        base = rep["dev_loo"]["A_current_rule"] + rep["val"]["A_current_rule"]
        wins = sum(a["f1_2s"] > b["f1_2s"] + 1e-9 for a, b in zip(rows, base))
        losses = sum(a["f1_2s"] < b["f1_2s"] - 1e-9 for a, b in zip(rows, base))
        ok = dv > base_dev and (vv is None or vv >= base_val) and wins > losses
        verdicts[m] = {"dev_loo_f1": dv, "val_f1": vv, "videos_better": wins, "videos_worse": losses, "accepted": bool(ok)}
    return {"baseline": {"dev_loo_f1": base_dev, "val_f1": base_val}, "methods": verdicts,
            "any_accepted": any(v["accepted"] for v in verdicts.values())}


def format_experiment(rep: dict, decision: dict) -> str:
    L = ["## Learned boundary model vs current SceneSeen", ""]
    names = {"dev_in_sample": "TRAINING fit on dev videos (in-sample; NOT a measure of accuracy)",
             "dev_loo": "Development: leave-one-video-out (each video scored by a model that never saw it)",
             "val": "Held-out VALIDATION videos (fitted/tuned on dev only)"}
    for part in ("dev_in_sample", "dev_loo", "val"):
        if not rep[part]["A_current_rule"]:
            continue
        L += [f"### {names[part]}", "",
              "| method | P@2s | R@2s | F1@1s | F1@2s | F1@3s | mean scene-count error | coverage | overflow |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            s = rep["summary"][part][m]
            L.append(f"| {m} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1_1s']:.3f} | {s['f1_2s']:.3f} | "
                     f"{s['f1_3s']:.3f} | {s['mean_abs_scene_count_error']:.2f} | {s['coverage']:.3f} | {s['overflow']:.3f} |")
        if part != "dev_in_sample":
            L += ["", "Per video, F1@2s (true → predicted scenes):", "", "| video | " + " | ".join(METHODS) + " |",
                  "|---|" + "---|" * len(METHODS)]
            for i, row in enumerate(rep[part]["A_current_rule"]):
                cells = [f"{rep[part][m][i]['f1_2s']:.2f} ({rep[part][m][i]['true_scenes']}→{rep[part][m][i]['pred_scenes']})"
                         for m in METHODS]
                L.append(f"| {row['video'][:38]} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["### Decision", ""]
    for m, v in decision["methods"].items():
        L.append(f"- **{m}**: dev-LOO F1 {v['dev_loo_f1']:.3f} (current {decision['baseline']['dev_loo_f1']:.3f}), "
                 f"validation F1 {v['val_f1'] if v['val_f1'] is None else round(v['val_f1'], 3)} "
                 f"(current {decision['baseline']['val_f1'] if decision['baseline']['val_f1'] is None else round(decision['baseline']['val_f1'], 3)}), "
                 f"better on {v['videos_better']} / worse on {v['videos_worse']} videos → "
                 f"{'ACCEPTED' if v['accepted'] else 'REJECTED'}")
    return "\n".join(L) + "\n"
