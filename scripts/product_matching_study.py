"""EXPERIMENTAL — DISABLED research (automatic product matching is not part of SceneSeen; docs/EXPERIMENTAL_MATCHING.md).

Calibration study for product matching (Phase 2B).

There is no licensed product catalogue yet, so the study uses the project's own footage:
detected object crops are the "products", and the question is whether an embedding puts two
crops of the SAME physical object (seen in different camera set-ups) closer together than crops
of different objects of the same type.

    python scripts/product_matching_study.py sample     # crops + blind pair sheets
    python scripts/product_matching_study.py evaluate   # compare embedders / signals, derive thresholds

* Within-video pairs are labelled by hand (S = same object, D = different, U = unsure):
  ground_truth/product_matching/pair_labels.json
* Cross-video pairs of the same type are different objects by construction (different productions).
Crops and sheets are written under data/ and reports/ and are never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sceneseen.commercial import analysis as A  # noqa: E402
from sceneseen.commercial import taxonomy as T  # noqa: E402
from sceneseen.commercial.frames import frame_name  # noqa: E402
from sceneseen.config import load_config  # noqa: E402
from sceneseen.grouping import boundaries_to_scenes  # noqa: E402
from sceneseen.media import VideoInfo, relocated  # noqa: E402
from sceneseen.pipeline import load_stage_outputs, unique_shots_for  # noqa: E402

GT = ROOT / "ground_truth" / "product_matching"
WORK = ROOT / "data" / "study" / "product_matching"
SHEETS = ROOT / "reports" / "product_matching" / "sheets"
PAD = 0.08


def crop_box(box, w, h, pad=PAD):
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, (x1 - px) * w)), int(max(0, (y1 - py) * h)), int(min(w, (x2 + px) * w)), int(min(h, (y2 + py) * h)))


def collect(cfg) -> list[dict]:
    """Every displayed detection of every analysed video, as one crop record."""
    from PIL import Image

    out = []
    for cdir in sorted(cfg.paths.cache_dir.glob("*/commercial")):
        c = cdir.parent
        if not (c / "result.json").exists() or not (c / "info.json").exists():
            continue
        info = json.loads((c / "info.json").read_text(encoding="utf-8"))
        up = json.loads((c / "upload.json").read_text(encoding="utf-8")) if (c / "upload.json").exists() else {}
        info["path"] = str(relocated(up.get("path", info["path"]), cfg.paths.uploads_dir))
        info = VideoInfo(**info)
        st = load_stage_outputs(cfg, info)
        res = json.loads((c / "result.json").read_text(encoding="utf-8"))
        corr = json.loads((c / "corrections.json").read_text(encoding="utf-8")) if (c / "corrections.json").exists() else None
        scenes = boundaries_to_scenes(corr["boundaries"] if corr else res["boundaries"], info.duration, st["shots"])
        uq = unique_shots_for(cfg, info, st, scenes)
        rec = A.analyze(st["cache"].dir, info, st["shots"], scenes, uq, cfg.commercial, allow_inference=False)
        fdir = cdir / f"frames_{cfg.commercial.frame_long_side}"
        for sc in rec["scenes"]:
            for cand in sc["candidates"]:
                if not cand["displayed"]:
                    continue
                need = cand["debug"]["threshold"]
                for d in cand["debug"]["detections"]:
                    if d["score"] < need:
                        continue
                    fp = fdir / frame_name(d["shot_id"], 0.5)
                    if not fp.exists():
                        continue
                    cid = hashlib.sha1(f"{c.name}|{d['shot_id']}|{cand['type_id']}|{d['box']}".encode()).hexdigest()[:12]
                    out.append({"id": cid, "video": c.name, "scene": sc["scene_id"], "shot": d["shot_id"],
                                "type": cand["type_id"], "score": d["score"], "box": d["box"], "frame": str(fp)})
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "crops").mkdir(exist_ok=True)
    seen = set()
    uniq = []
    for r in out:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        p = WORK / "crops" / f"{r['id']}.jpg"
        if not p.exists():
            with Image.open(r["frame"]) as im:
                im = im.convert("RGB")
                im.crop(crop_box(r["box"], *im.size)).save(p, quality=92)
        with Image.open(p) as im:
            r["w"], r["h"] = im.size
        r.pop("frame")
        uniq.append(r)
    (WORK / "crops.json").write_text(json.dumps(uniq), encoding="utf-8")
    return uniq


def cmd_sample(cfg) -> None:
    from PIL import Image, ImageDraw

    crops = collect(cfg)
    rng = random.Random(20261005)
    by: dict[tuple[str, str], list[dict]] = {}
    for r in crops:
        if min(r["w"], r["h"]) >= 40:
            by.setdefault((r["video"], r["type"]), []).append(r)
    pairs = []
    for (video, typ), rows in sorted(by.items()):
        cand = [(a, b) for i, a in enumerate(rows) for b in rows[i + 1:] if a["shot"] != b["shot"]]
        rng.shuffle(cand)
        same_scene = [p for p in cand if p[0]["scene"] == p[1]["scene"]][:5]
        other = [p for p in cand if p[0]["scene"] != p[1]["scene"]][:3]
        pairs += same_scene + other
    rng.shuffle(pairs)
    pairs = pairs[:288]
    GT.mkdir(parents=True, exist_ok=True)
    (GT / "pairs.json").write_text(json.dumps({
        "description": "Within-video pairs of detected object crops of the same type from different shots. "
                       "Crop ids are sha1(video id | shot | type | box).",
        "pairs": [{"id": f"q{n:03d}", "a": a["id"], "b": b["id"], "type": a["type"], "video": a["video"],
                   "same_scene": a["scene"] == b["scene"]} for n, (a, b) in enumerate(pairs)]}, indent=1) + "\n",
        encoding="utf-8")
    SHEETS.mkdir(parents=True, exist_ok=True)
    cw, ch, cols, rows_ = 132, 132, 6, 6
    for s0 in range(0, len(pairs), cols * rows_):
        sheet = Image.new("RGB", (cols * (2 * cw + 14), rows_ * (ch + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (a, b) in enumerate(pairs[s0:s0 + cols * rows_]):
            x, y = (k % cols) * (2 * cw + 14), (k // cols) * (ch + 16)
            d.text((x + 2, y + 1), f"q{s0 + k:03d} {a['type']}", fill="black")
            for j, r in enumerate((a, b)):
                im = Image.open(WORK / "crops" / f"{r['id']}.jpg")
                im.thumbnail((cw - 2, ch - 2))
                sheet.paste(im, (x + j * cw, y + 14))
        sheet.save(SHEETS / f"sheet_{s0 // (cols * rows_):02d}.jpg", quality=88)
    types = {}
    for a, _ in pairs:
        types[a["type"]] = types.get(a["type"], 0) + 1
    print(f"{len(crops)} crops from {len({r['video'] for r in crops})} videos; {len(pairs)} pairs to label; by type: {types}")


def _hsv_hist(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = img[int(h * 0.15):int(h * 0.85) or 1, int(w * 0.15):int(w * 0.85) or 1]
    hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return (hist / max(hist.sum(), 1e-9)).astype(np.float32)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def cmd_evaluate(cfg, models: list[str]) -> None:
    from PIL import Image

    from sceneseen.experimental.matching.embedder import EmbedderUnavailable, get_embedder

    crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
    idx = {r["id"]: i for i, r in enumerate(crops)}
    pairs = json.loads((GT / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    labels = json.loads((GT / "pair_labels.json").read_text(encoding="utf-8"))["labels"]
    S = [(idx[p["a"]], idx[p["b"]]) for p in pairs if labels.get(p["id"]) == "S" and p["a"] in idx and p["b"] in idx]
    Dn = [(idx[p["a"]], idx[p["b"]]) for p in pairs if labels.get(p["id"]) == "D" and p["a"] in idx and p["b"] in idx]
    video = np.array([r["video"] for r in crops])
    typ = np.array([r["type"] for r in crops])
    size = np.array([min(r["w"], r["h"]) for r in crops])
    compat = {t: T.compatible_types(t) for t in T.OBJECTS}
    hist = np.stack([_hsv_hist(WORK / "crops" / f"{r['id']}.jpg") for r in crops])
    CM = np.zeros((len(crops), len(crops)), np.float32)                 # colour-histogram intersection, all pairs
    for i0 in range(0, len(crops), 64):
        CM[i0:i0 + 64] = np.minimum(hist[i0:i0 + 64, None, :], hist[None, :, :]).sum(axis=-1)
    color = lambda i, j: float(CM[i, j])  # noqa: E731
    sims: dict[str, np.ndarray] = {}
    print(f"{len(crops)} crops, {len(S)} same-object pairs, {len(Dn)} different-object pairs (same video, same type)\n")
    report = {"crops": len(crops), "same_pairs": len(S), "different_pairs": len(Dn), "models": {}}
    queries = [(a, b) for a, b in S] + [(b, a) for a, b in S]          # (query, its true reference)
    for name in models:
        cache = WORK / f"emb_{name}.npy"
        try:
            if cache.exists():
                E = np.load(cache)
                secs = None
            else:
                emb = get_embedder(name)
                import time

                t = time.perf_counter()
                E = emb.embed([np.asarray(Image.open(WORK / "crops" / f"{r['id']}.jpg").convert("RGB")) for r in crops])
                secs = round((time.perf_counter() - t) / len(crops) * 1000, 2)
                np.save(cache, E)
        except EmbedderUnavailable as e:
            print(f"{name}: unavailable ({e})")
            continue
        sim = E @ E.T
        sims[name] = sim
        pos = np.array([sim[a, b] for a, b in S])
        neg = np.array([sim[a, b] for a, b in Dn])

        def retrieval(gated: bool, score=None):
            score = sim if score is None else score
            top1 = top3 = 0
            present, absent, correct = [], [], []
            for q, ref in queries:
                allowed = np.isin(typ, list(compat[typ[q]])) if gated else np.ones(len(crops), bool)
                gal = np.where((video != video[q]) & allowed)[0]          # other productions: different objects
                d = score[q, gal]
                best_d = float(d.max()) if len(d) else -1.0
                s_ref = float(score[q, ref])
                rank = 1 + int((d > s_ref).sum())
                top1 += rank == 1
                top3 += rank <= 3
                present.append(max(s_ref, best_d))                      # top-1 score when the product IS in the catalogue
                correct.append(rank == 1)
                absent.append(best_d)                                   # top-1 score when it is NOT
            return top1 / len(queries), top3 / len(queries), np.array(present), np.array(correct), np.array(absent)

        g1, g3, present, correct, absent = retrieval(True)
        u1, u3, *_ = retrieval(False)
        # colour inside retrieval: score = cosine + w * colour, w chosen on pairs from OTHER videos
        c1_by_w = {}
        for w in (0.0, 0.1, 0.2, 0.3, 0.5):
            c1_by_w[w] = retrieval(True, sim + w * CM)[0]
        lov = []
        qv = np.array([video[q] for q, _ in queries])
        for v in sorted(set(qv)):
            tr = qv != v
            def acc(w, mask):
                sc = sim + w * CM
                hits = []
                for (q, ref), use in zip(queries, mask):
                    if not use:
                        continue
                    gal = np.where((video != video[q]) & np.isin(typ, list(compat[typ[q]])))[0]
                    hits.append(not (sc[q, gal] > sc[q, ref]).any())
                return float(np.mean(hits)) if hits else 0.0
            w_star = max((0.0, 0.1, 0.2, 0.3, 0.5), key=lambda w: acc(w, tr))
            lov.append((acc(w_star, ~tr), int((~tr).sum())))
        top1_colour_lovo = float(sum(a * n for a, n in lov) / sum(n for _, n in lov))
        # colour as an extra signal: logistic combination, leave-one-video-out
        from sklearn.linear_model import LogisticRegression

        Xp = np.array([[sim[a, b], color(a, b)] for a, b in S])
        Xn = np.array([[sim[a, b], color(a, b)] for a, b in Dn])
        X, y = np.r_[Xp, Xn], np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
        vids = np.array([video[a] for a, _ in S] + [video[a] for a, _ in Dn])
        comb = np.zeros(len(y))
        for v in set(vids):
            tr = vids != v
            if len(set(y[tr])) == 2:
                comb[~tr] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
        m = {"pair_auc": _auc(pos, neg), "pair_auc_with_colour": _auc(comb[y == 1], comb[y == 0]),
             "colour_only_auc": _auc(Xp[:, 1], Xn[:, 1]),
             "top1_gated": g1, "top3_gated": g3, "top1_ungated": u1, "top3_ungated": u3,
             "top1_gated_plus_colour_lovo": top1_colour_lovo, "top1_by_colour_weight": c1_by_w,
             "same_mean": float(pos.mean()), "different_mean": float(neg.mean()), "dim": int(E.shape[1]),
             "ms_per_crop": secs,
             "gallery_size_gated_mean": float(np.mean([((video != video[q]) & np.isin(typ, list(compat[typ[q]]))).sum() for q, _ in queries]))}
        # thresholds: precision of "accept top-1 at score >= t" when half the queries have no true product in the catalogue
        sweep = []
        for t in np.round(np.arange(0.30, 0.96, 0.01), 2):
            acc_p = present >= t
            acc_a = absent >= t
            tp = int((acc_p & correct).sum())
            fp = int((acc_p & ~correct).sum() + acc_a.sum())
            sweep.append({"t": float(t), "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / len(present), "false_accept_absent": float(acc_a.mean())})
        m["sweep"] = sweep
        m["present"] = present.round(4).tolist()
        m["correct"] = correct.astype(int).tolist()
        m["absent"] = absent.round(4).tolist()
        m["by_size"] = {}
        for lo, hi in ((0, 80), (80, 160), (160, 10000)):
            sel = [k for k, (q, r) in enumerate(queries) if lo <= min(size[q], size[r]) < hi]
            if sel:
                m["by_size"][f"{lo}-{hi}px"] = {"n": len(sel), "top1": float(np.mean(correct[sel]))}
        report["models"][name] = m
        print(f"{name:20s} dim {E.shape[1]:4d} | pair AUC {m['pair_auc']:.3f} (+colour {m['pair_auc_with_colour']:.3f}) | "
              f"top-1 gated {g1:.3f} / ungated {u1:.3f} | +colour (held-out weight) {top1_colour_lovo:.3f} | "
              f"top-3 gated {g3:.3f}" + (f" | {secs} ms/crop" if secs else ""))
    # ---- ensembles of two embedders (mean of z-scored similarities)
    z = {k: (v - v.mean()) / v.std() for k, v in sims.items()}
    names = list(sims)
    report["ensembles"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sc = (z[a] + z[b]) / 2
            hits = []
            for q, ref in queries:
                gal = np.where((video != video[q]) & np.isin(typ, list(compat[typ[q]])))[0]
                hits.append(not (sc[q, gal] > sc[q, ref]).any())
            auc = _auc(np.array([sc[x, y_] for x, y_ in S]), np.array([sc[x, y_] for x, y_ in Dn]))
            report["ensembles"][f"{a}+{b}"] = {"top1_gated": float(np.mean(hits)), "pair_auc": auc}
            print(f"  ensemble {a}+{b}: top-1 gated {np.mean(hits):.3f} | pair AUC {auc:.3f}")
    # ---- multi-occurrence evidence: objects with >= 3 labelled-same crops (union of S links)
    parent = list(range(len(crops)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in S:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a, b in S:
        for x in (a, b):
            groups.setdefault(find(x), [])
            if x not in groups[find(x)]:
                groups[find(x)].append(x)
    multi = [g for g in groups.values() if len(g) >= 3]
    report["multi_occurrence"] = {"objects_with_3plus_views": len(multi)}
    for name, sim in sims.items():
        single, agg = [], []
        for g in multi:
            for ref in g:
                qs = [x for x in g if x != ref]
                gal = np.where((video != video[ref]) & np.isin(typ, list(compat[typ[ref]])))[0]
                for q in qs:                                             # one occurrence at a time
                    single.append(not (sim[q, gal] > sim[q, ref]).any())
                mean_ref = np.mean([sim[q, ref] for q in qs])            # all occurrences together
                mean_gal = np.mean([sim[q, gal] for q in qs], axis=0)
                agg.append(not (mean_gal > mean_ref).any())
        if single:
            report["multi_occurrence"][name] = {"top1_single_occurrence": float(np.mean(single)),
                                                "top1_mean_over_occurrences": float(np.mean(agg)), "trials": len(agg)}
            print(f"  multi-occurrence {name}: single {np.mean(single):.3f} -> mean over occurrences {np.mean(agg):.3f} "
                  f"({len(multi)} objects, {len(agg)} trials)")
    out = ROOT / "reports" / "product_matching"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_study.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def cmd_calibrate(cfg, name: str) -> None:
    """Thresholds and confidence curve for the shipped matcher, measured with the real index code.

    Simulated catalogue: each labelled object contributes ONE reference crop; every crop of a
    compatible type from other productions is a different product. Each object is tested twice:
    with its reference in the catalogue (the right answer exists) and without (the right answer
    is "unknown product"). Thresholds are picked on the other videos and scored on the held-out one.
    """
    from sklearn.linear_model import LogisticRegression

    from sceneseen.experimental.matching.index import CatalogIndex
    from sceneseen.experimental.matching.signals import crop_weight

    crops = json.loads((WORK / "crops.json").read_text(encoding="utf-8"))
    idx = {r["id"]: i for i, r in enumerate(crops)}
    pairs = json.loads((GT / "pairs.json").read_text(encoding="utf-8"))["pairs"]
    labels = json.loads((GT / "pair_labels.json").read_text(encoding="utf-8"))["labels"]
    S = [(idx[p["a"]], idx[p["b"]]) for p in pairs if labels.get(p["id"]) == "S"]
    E = np.load(WORK / f"emb_{name}.npy")
    hist = np.stack([_hsv_hist(WORK / "crops" / f"{r['id']}.jpg") for r in crops])
    video = np.array([r["video"] for r in crops])
    typ = np.array([r["type"] for r in crops])
    side = np.array([min(r["w"], r["h"]) for r in crops], np.float32)
    cat = np.array([T.OBJECTS[t].category for t in typ])
    parent = list(range(len(crops)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in S:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a, b in S:
        for x in (a, b):
            g = groups.setdefault(find(x), [])
            if x not in g:
                g.append(x)

    def index_for(rows: np.ndarray) -> CatalogIndex:
        rows = np.sort(rows)
        ix = CatalogIndex(name)
        ix.E, ix.C, ix.product, ix.image = E[rows], hist[rows], rows.astype(np.int64), rows.astype(np.int64)
        ix._type_arr, ix._cat_arr = typ[rows], cat[rows]
        ix.types = {int(r): typ[r] for r in rows}
        return ix

    def trials(w: float, max_occ: int):
        """One row per (object, reference, scenario): top score, margin, correct?, video, n occurrences."""
        out = []
        for g in groups.values():
            v = video[g[0]]
            others = np.where(video != v)[0]
            for ref in g:
                qs = [x for x in g if x != ref][:max_occ]
                Q, Qc, wt = E[qs], hist[qs], np.array([crop_weight(side[q]) for q in qs])
                for present in (True, False):
                    rows = np.r_[others, [ref]] if present else others
                    res = index_for(rows).search(Q, Qc, wt, typ[ref], w, top_k=2, shortlist=200)["results"]
                    if not res:
                        continue
                    margin = res[0]["score"] - res[1]["score"] if len(res) > 1 else res[0]["score"]
                    out.append((res[0]["score"], margin, present and res[0]["product_id"] == ref, present, v, len(qs),
                                float(np.mean(side[qs]))))
        return out

    print(f"{len(groups)} labelled objects with >= 2 views; embedder {name}\n")
    print("colour weight | top-1 when present (1 occurrence) | top-1 when present (up to 5 occurrences)")
    acc = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        a1 = np.mean([t[2] for t in trials(w, 1) if t[3]])
        a5 = np.mean([t[2] for t in trials(w, 5) if t[3]])
        acc[w] = (float(a1), float(a5))
        print(f"   {w:4.2f}       |            {a1:.3f}                 |            {a5:.3f}")
    # choose the colour weight on other videos, report the held-out accuracy
    vids = sorted(set(video[[g[0] for g in groups.values()]]))
    cache = {w: trials(w, 5) for w in acc}
    held = []
    for v in vids:
        best = max(acc, key=lambda w: np.mean([t[2] for t in cache[w] if t[3] and t[4] != v] or [0]))
        held += [t[2] for t in cache[best] if t[3] and t[4] == v]
    w_star = max(acc, key=lambda w: acc[w][1])
    print(f"\nchosen colour weight: {w_star} (held-out top-1 with the weight picked on other videos: {np.mean(held):.3f})")
    T_ = cache[w_star]
    score = np.array([t[0] for t in T_])
    margin = np.array([t[1] for t in T_])
    ok = np.array([t[2] for t in T_])
    present = np.array([t[3] for t in T_])
    tv = np.array([t[4] for t in T_])
    # Which signal says "this top candidate is really the product"?
    from sklearn.metrics import roc_auc_score

    y = ok.astype(int)
    auc_score, auc_margin = float(roc_auc_score(y, score)), float(roc_auc_score(y, margin))
    X = np.stack([score, margin], axis=1)
    held_p = np.zeros(len(y))
    def fit(Xtr, ytr):
        """Logistic regression on standardised inputs, returned as weights on the raw inputs."""
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
        w = m.coef_[0] / sd
        return w, float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum())

    for v in vids:                                       # leave-one-video-out probabilities
        tr = tv != v
        w, b0 = fit(X[tr], y[tr])
        held_p[~tr] = 1 / (1 + np.exp(-(X[~tr] @ w + b0)))
    w, bias = fit(X, y)
    w_score, w_margin = float(w[0]), float(w[1])
    print(f"what predicts a correct top candidate?  AUC: raw score {auc_score:.3f} | lead over runner-up {auc_margin:.3f} | "
          f"both (held-out) {roc_auc_score(y, held_p):.3f}")
    print(f"match confidence = sigmoid({w_score:.3f} * score + {w_margin:.3f} * margin + {bias:.3f})")
    cuts = {"high_confidence": 0.80, "possible_match": 0.50, "uncertain": 0.25}
    bands = [("high_confidence", 0.80, 1.01), ("possible_match", 0.50, 0.80), ("uncertain", 0.25, 0.50),
             ("no_reliable_match", -1.0, 0.25)]
    print("\nheld-out behaviour of the states (confidence model fitted on the other videos):")
    rep_states = {}
    for st, lo, hi in bands:
        m = (held_p >= lo) & (held_p < hi)
        n, c = int(m.sum()), int(ok[m].sum())
        rep_states[st] = {"n": n, "correct": c, "precision": c / n if n else None,
                          "product_absent": int((~present[m]).sum())}
        print(f"  {st:18s}: {n:4d} cases, top candidate correct in {c} ({(c / n if n else 0):.1%}); "
              f"{int((~present[m]).sum())} had NO true product in the catalogue")
    n_abs, n_pre = int((~present).sum()), int(present.sum())
    abs_high = int(((held_p >= 0.8) & ~present).sum())
    print(f"\nproduct NOT in the catalogue ({n_abs} cases): wrongly shown as high confidence {abs_high} "
          f"({abs_high / n_abs:.1%}); shown as possible {int(((held_p >= 0.5) & (held_p < 0.8) & ~present).sum())}; "
          f"uncertain or no match {int(((held_p < 0.5) & ~present).sum())} ({((held_p < 0.5) & ~present).sum() / n_abs:.1%})")
    found_high = int(((held_p >= 0.8) & ok).sum())
    print(f"product IN the catalogue ({n_pre} cases): ranked first {int(ok.sum())} ({ok.sum() / n_pre:.1%}); "
          f"ranked first AND high confidence {found_high} ({found_high / n_pre:.1%})")
    # Score a WRONG product typically reaches as the top candidate (75th percentile when the true
    # product is absent). The matcher uses it as the runner-up when a catalogue has too few
    # comparable products for a real one ([matching] background_score).
    background = float(np.percentile(score[~present], 75))
    print(f"background score (75th percentile of the best WRONG candidate): {background:.3f}")
    out = ROOT / "reports" / "product_matching"
    (out / ("calibration.json" if name == "openclip_b32" else f"calibration_{name}.json")).write_text(json.dumps({
        "embedder": name, "objects": len(groups), "trials": len(score), "colour_weight": w_star,
        "top1_by_colour_weight": {str(k): {"one_occurrence": v[0], "up_to_5_occurrences": v[1]} for k, v in acc.items()},
        "held_out_top1": float(np.mean(held)),
        "auc_correct_top_candidate": {"raw_score": auc_score, "margin": auc_margin,
                                      "score_and_margin_held_out": float(roc_auc_score(y, held_p))},
        "confidence_model": {"w_score": w_score, "w_margin": w_margin, "bias": bias},
        "background_score": background,
        "state_cuts": cuts, "held_out_states": rep_states,
        "product_absent": {"cases": n_abs, "shown_high_confidence": abs_high},
        "product_present": {"cases": n_pre, "ranked_first": int(ok.sum()), "ranked_first_and_high_confidence": found_high}},
        indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "evaluate", "calibrate"])
    ap.add_argument("--models", default="openclip_b32,dinov2_small,dinov2_base,marqo_ecommerce_b,siglip_base")
    ap.add_argument("--embedder", default="openclip_b32")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "sample":
        cmd_sample(cfg)
    elif a.cmd == "evaluate":
        cmd_evaluate(cfg, a.models.split(","))
    else:
        cmd_calibrate(cfg, a.embedder)
