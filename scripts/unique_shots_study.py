"""Calibration study for Unique Shots (repeated-shot detection).

    python scripts/unique_shots_study.py sample     # sample within-scene shot pairs + blind contact sheets
    python scripts/unique_shots_study.py evaluate   # compare similarity methods on the labelled pairs

`sample` writes  ground_truth/unique_shots/pairs.json  (pairs, no labels) and blind contact sheets
(shuffled, no scores shown) to reports/unique_shots/sheets/ (images are never committed).
Labels (D = same camera set-up / repeated shot, N = different shot, U = unsure) are stored in
ground_truth/unique_shots/pair_labels.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sceneseen import similarity as sim  # noqa: E402
from sceneseen.benchmark import load_dataset  # noqa: E402
from sceneseen.config import load_config  # noqa: E402
from sceneseen.grouping import boundaries_to_scenes  # noqa: E402

OUT = ROOT / "ground_truth" / "unique_shots"
SHEETS = ROOT / "reports" / "unique_shots" / "sheets"
BINS = [0.0, 0.70, 0.78, 0.84, 0.88, 0.91, 0.94, 0.97, 1.01]
PER_BIN = 24
MAX_GAP_SHOTS = 40


def thumbs_gray(item: dict) -> list[np.ndarray | None]:
    import cv2

    out = []
    for s in item["stage"]["shots"]:
        p = item["stage"]["thumbs_dir"] / f"shot_{s.index:04d}.jpg"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None
        out.append(img)
    return out


def measures(item: dict) -> dict[str, np.ndarray]:
    """All candidate similarity matrices for one video."""
    import cv2

    f = item["stage"]["features"]
    fs = sim.clip_frame_similarity(f["clip"])
    out = {
        "clip_mean": sim.clip_mean_similarity(f["clip"]),
        "clip_best_frame": sim.clip_best_frame_similarity(fs),
        "clip_matched": sim.clip_matched_similarity(fs),
        "clip_worst_matched": sim.clip_worst_matched_similarity(fs),
        "color": sim.color_similarity(f["color"]),
    }
    grays = thumbs_gray(item)
    n = len(grays)
    blank = np.zeros((72, 128), np.uint8)
    small = [cv2.resize(g, (128, 72), interpolation=cv2.INTER_AREA) if g is not None else blank for g in grays]
    out["phash"] = sim.hash_similarity(np.stack([sim.phash_bits(g) for g in small]))
    out["_small"] = small  # SSIM is computed lazily per pair (O(pairs), not O(N^2))
    out["_n"] = n
    return out


def scene_of_shots(item: dict) -> np.ndarray:
    shots = item["stage"]["shots"]
    scenes = boundaries_to_scenes(item["gt"], item["info"].duration, shots)
    sid = np.zeros(len(shots), int)
    for k, sc in enumerate(scenes):
        if sc["shot_start"] is not None:
            sid[sc["shot_start"]:sc["shot_end"] + 1] = k
    return sid


def cmd_sample(cfg) -> None:
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(20261003)
    data = load_dataset(cfg, "all")
    cands = []
    for it in data:
        m = measures(it)
        sid = scene_of_shots(it)
        n = m["_n"]
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + MAX_GAP_SHOTS)):
                if sid[i] == sid[j]:
                    cands.append((it["stem"], i, j, float(m["clip_matched"][i, j])))
    print(f"{len(cands)} within-scene pairs in {len(data)} videos")
    chosen = []
    vals = np.array([c[3] for c in cands])
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        idx = np.where((vals >= lo) & (vals < hi))[0]
        # round-robin over videos so no single film dominates a bin
        by_video: dict[str, list[int]] = {}
        for k in rng.permutation(idx):
            by_video.setdefault(cands[k][0], []).append(int(k))
        picked = []
        while len(picked) < PER_BIN and any(by_video.values()):
            for v in sorted(by_video):
                if by_video[v] and len(picked) < PER_BIN:
                    picked.append(by_video[v].pop())
        chosen += picked
        print(f"  clip_matched [{lo:.2f}, {hi:.2f}): {len(idx):6d} pairs -> sampled {len(picked)}")
    order = rng.permutation(len(chosen))
    pairs = [{"id": f"p{n:03d}", "video": cands[chosen[k]][0], "shot_a": cands[chosen[k]][1],
              "shot_b": cands[chosen[k]][2]} for n, k in enumerate(order)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pairs.json").write_text(json.dumps({
        "description": "Within-scene shot pairs sampled evenly across the CLIP multi-frame similarity range "
                       "(shuffled). Shot indices refer to the shot list of the default shot detector config.",
        "sampling_seed": 20261003, "pairs": pairs}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # blind contact sheets: id + two thumbnails, no scores
    SHEETS.mkdir(parents=True, exist_ok=True)
    thumbs = {it["stem"]: it["stage"]["thumbs_dir"] for it in data}
    tw, th, cols, rows = 196, 110, 4, 8
    for s0 in range(0, len(pairs), cols * rows):
        sheet = Image.new("RGB", (cols * (2 * tw + 14), rows * (th + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for k, p in enumerate(pairs[s0:s0 + cols * rows]):
            x, y = (k % cols) * (2 * tw + 14), (k // cols) * (th + 18)
            d.text((x + 2, y + 2), p["id"], fill="black")
            for q, key in enumerate(("shot_a", "shot_b")):
                tp = thumbs[p["video"]] / f"shot_{p[key]:04d}.jpg"
                if tp.exists():
                    sheet.paste(Image.open(tp).resize((tw, th)), (x + q * tw + (2 if q else 0), y + 16))
        sheet.save(SHEETS / f"sheet_{s0 // (cols * rows):02d}.jpg", quality=85)
    print(f"wrote {len(pairs)} pairs and {len(list(SHEETS.glob('*.jpg')))} sheets")


def pair_values(cfg) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    """(pairs, {measure: values}, labels) for every labelled pair (D=1, N=0; U excluded)."""
    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    labels = json.loads((OUT / "pair_labels.json").read_text(encoding="utf-8"))["labels"]
    data = {it["stem"]: it for it in load_dataset(cfg, "all")}
    cache: dict[str, dict] = {}
    rows, y = [], []
    vals: dict[str, list[float]] = {}
    for p in pairs:
        lab = labels.get(p["id"])
        if lab not in ("D", "N"):
            continue
        if p["video"] not in cache:
            cache[p["video"]] = measures(data[p["video"]])
        m = cache[p["video"]]
        i, j = p["shot_a"], p["shot_b"]
        for k in ("clip_mean", "clip_best_frame", "clip_matched", "clip_worst_matched", "color", "phash"):
            vals.setdefault(k, []).append(float(m[k][i, j]))
        vals.setdefault("ssim", []).append(sim.ssim(m["_small"][i], m["_small"][j]))
        rows.append(p)
        y.append(1 if lab == "D" else 0)
    return rows, {k: np.array(v) for k, v in vals.items()}, np.array(y)


def best_threshold(v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 for 'duplicate' (candidates = midpoints between sorted values)."""
    order = np.sort(np.unique(v))
    cands = (order[:-1] + order[1:]) / 2 if len(order) > 1 else order
    best = (0.0, -1.0)
    for t in cands:
        pred = v >= t
        tp = int((pred & (y == 1)).sum())
        if tp == 0:
            continue
        p, r = tp / pred.sum(), tp / (y == 1).sum()
        f = 2 * p * r / (p + r)
        if f > best[1]:
            best = (float(t), f)
    return best


def cmd_evaluate(cfg) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows, vals, y = pair_values(cfg)
    videos = np.array([r["video"] for r in rows])
    combos = {
        "clip_matched+color": ["clip_matched", "color"],
        "clip_matched+phash": ["clip_matched", "phash"],
        "clip_matched+ssim": ["clip_matched", "ssim"],
        "clip_matched+color+phash+ssim": ["clip_matched", "color", "phash", "ssim"],
    }
    print(f"{len(y)} labelled pairs: {int(y.sum())} duplicate, {int((1 - y).sum())} different, "
          f"{len(set(videos))} videos\n")
    print(f"{'method':32s} {'ROC-AUC':>8s} {'AP':>6s} {'bestF1':>7s} {'thr':>6s} {'held-out F1 (leave-one-video-out)':>34s}")
    report = {}
    for name in ("clip_mean", "clip_best_frame", "clip_matched", "clip_worst_matched", "color", "phash", "ssim",
                 *combos):
        if name in combos:  # logistic combination; scores for AUC come from leave-one-video-out predictions
            X = np.stack([vals[k] for k in combos[name]], axis=1)
            v = np.zeros(len(y))
            for vid in set(videos):
                tr = videos != vid
                if len(set(y[tr])) < 2:
                    continue
                v[~tr] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
        else:
            v = vals[name]
        thr, f1 = best_threshold(v, y)
        # held-out: choose the threshold on the other videos, score on the left-out video's pairs
        tp = fp = fn = 0
        for vid in set(videos):
            tr = videos != vid
            t, _ = best_threshold(v[tr], y[tr])
            pred = v[~tr] >= t
            tp += int((pred & (y[~tr] == 1)).sum())
            fp += int((pred & (y[~tr] == 0)).sum())
            fn += int((~pred & (y[~tr] == 1)).sum())
        ho = 2 * tp / max(1, 2 * tp + fp + fn)
        report[name] = {"roc_auc": float(roc_auc_score(y, v)), "average_precision": float(average_precision_score(y, v)),
                        "best_f1": f1, "best_threshold": thr, "held_out_f1": ho}
        print(f"{name:32s} {report[name]['roc_auc']:8.3f} {report[name]['average_precision']:6.3f} {f1:7.3f} "
              f"{thr:6.3f} {ho:34.3f}")
    # three-state thresholds for the chosen measure
    v = vals["clip_matched"]
    hi = min((t for t in np.sort(np.unique(v)) if ((v >= t) & (y == 0)).sum() == 0), default=1.0)
    lo = max((t for t in np.sort(np.unique(v)) if ((v < t) & (y == 1)).sum() == 0), default=0.0)
    print(f"\nclip_matched: no labelled 'different' pair at or above {hi:.3f}; no labelled 'duplicate' below {lo:.3f}")
    for name in ("clip_matched",):
        print("\nvalue distribution (clip_matched):")
        for a, b in zip(BINS[:-1], BINS[1:]):
            m_ = (v >= a) & (v < b)
            print(f"  [{a:.2f}, {b:.2f}): {int((y[m_] == 1).sum()):3d} duplicate  {int((y[m_] == 0).sum()):3d} different")
    out = ROOT / "reports" / "unique_shots"
    out.mkdir(parents=True, exist_ok=True)
    (out / "method_comparison.json").write_text(json.dumps(
        {"n_pairs": int(len(y)), "n_duplicate": int(y.sum()), "methods": report,
         "always_duplicate_at_or_above": float(hi), "never_duplicate_below": float(lo)}, indent=1) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "evaluate"])
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args()
    {"sample": cmd_sample, "evaluate": cmd_evaluate}[a.cmd](load_config(*a.config))
