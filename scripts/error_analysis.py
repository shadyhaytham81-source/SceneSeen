"""Per-error diagnosis of SceneSeen predictions against human labels.

For every false boundary (FP) and missed boundary (FN) at ±2 s it reports:
  * the nearest detected shot cut, its boundary score / depth / grouping decision
  * the transition type at that cut (hard cut vs gradual, from TransNetV2 probabilities)
  * the local editing pace (cuts per 10 s), which flags fast-cut sequences
  * a contact strip (shots before | shots after) written to reports/errors/ for visual review

    python scripts/error_analysis.py [--config extra.toml] [--out reports/errors]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sceneseen.benchmark import load_dataset, predict  # noqa: E402
from sceneseen.config import load_config  # noqa: E402
from sceneseen.evaluation import match_boundaries  # noqa: E402
from sceneseen import grouping  # noqa: E402

TOL = 2.0


def transition_info(shots_rec: dict, t: float, fps: float) -> dict:
    probs = shots_rec.get("transition_probs")
    if not probs:
        return {"transition": "unknown"}
    p = np.asarray(probs)
    f = int(round(t * fps))
    lo, hi = max(0, f - 12), min(len(p), f + 12)
    above = int((p[lo:hi] > 0.5).sum())
    peak = float(p[lo:hi].max()) if hi > lo else 0.0
    soft = int((p[lo:hi] > 0.1).sum())
    kind = "hard cut" if above == 1 else ("gradual (dissolve/fade)" if above > 1 else
                                          ("weak/undetected transition" if soft else "no transition"))
    return {"transition": kind, "frames_above_0.5": above, "peak_prob": round(peak, 3)}


def pace(cut_times: np.ndarray, t: float, half: float = 10.0) -> float:
    return float(((cut_times > t - half) & (cut_times < t + half)).sum()) * 10.0 / (2 * half)


def strip(stage: dict, cut_index: int | None, t: float, out: Path, title: str, n: int = 4) -> None:
    from PIL import Image, ImageDraw

    shots = stage["shots"]
    if cut_index is None:  # no shot cut: show the shot containing t and neighbours
        cut_index = max(0, min(len(shots) - 2, next((s.index for s in shots if s.start <= t < s.end), 0)))
    idx = list(range(max(0, cut_index - n + 1), min(len(shots), cut_index + 1 + n)))
    tw, th = 192, 108
    img = Image.new("RGB", (len(idx) * tw + 12, th + 34), "white")
    d = ImageDraw.Draw(img)
    d.text((4, 2), title[:150], fill="black")
    x = 0
    for i in idx:
        p = stage["thumbs_dir"] / f"shot_{i:04d}.jpg"
        if i == cut_index + 1:
            d.rectangle([x, 14, x + 9, th + 30], fill=(230, 40, 40))
            x += 12
        if p.exists():
            img.paste(Image.open(p).resize((tw - 4, th)), (x, 16))
        d.text((x + 2, th + 18), f"#{i} {shots[i].start:.1f}s", fill="black")
        x += tw
    img.save(out, quality=82)


def analyse(cfg, out_dir: Path) -> dict:
    data = load_dataset(cfg, "all")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for it in data:
        stage, info, gt = it["stage"], it["info"], it["gt"]
        g = grouping.group_shots(stage["shots"], stage["features"]["clip"], stage["features"]["color"],
                                 info.duration, cfg.grouping)
        pred = g["boundaries"]
        cuts = g["cuts"]
        cut_times = np.array([c["time"] for c in cuts])
        m = match_boundaries(pred, gt, TOL)
        mp, mg = {a for a, _, _ in m}, {b for _, b, _ in m}
        errs = []
        for kind, times, matched in (("FP", pred, mp), ("FN", gt, mg)):
            for k, t in enumerate(times):
                if k in matched:
                    continue
                j = int(np.argmin(np.abs(cut_times - t))) if len(cut_times) else None
                near = cuts[j] if j is not None else None
                dist = abs(near["time"] - t) if near else None
                e = {"type": kind, "time": round(t, 2),
                     "nearest_cut_dist": round(dist, 2) if dist is not None else None,
                     "cut_score": near and near["score"], "depth": near and near["depth"],
                     "decision": near and near["decision"],
                     "pace_cuts_per_10s": round(pace(cut_times, t), 1),
                     **transition_info(stage["shots_rec"], t, info.fps)}
                name = f"{it['stem'][:24].strip()}_{kind}_{t:07.1f}.jpg".replace("/", "_")
                strip(stage, near["cut"] if near and dist <= TOL else None, t, out_dir / name,
                      f"{it['stem'][:40]} {kind} @ {t:.1f}s  score={e['cut_score']} depth={e['depth']} "
                      f"{e['decision']} | {e['transition']}")
                e["strip"] = name
                errs.append(e)
        report[it["stem"]] = {"pred": pred, "gt": gt, "errors": sorted(errs, key=lambda e: e["time"])}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--out", default=str(ROOT / "reports/errors"))
    a = ap.parse_args()
    cfg = load_config(*a.config)
    rep = analyse(cfg, Path(a.out))
    (Path(a.out) / "errors.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    for stem, r in rep.items():
        print(f"\n== {stem[:60]}  pred={len(r['pred'])} gt={len(r['gt'])}")
        for e in r["errors"]:
            print(f"  {e['type']} {e['time']:8.1f}s  cut±{e['nearest_cut_dist']}  score={e['cut_score']} "
                  f"depth={e['depth']} {e['decision']:<12} {e['transition']:<26} pace={e['pace_cuts_per_10s']}")


if __name__ == "__main__":
    main()
