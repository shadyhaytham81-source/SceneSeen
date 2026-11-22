"""Product matching evaluation: whole-photo embeddings vs garment crops vs improved ranking.

Two parts, both on the REAL local catalogue and SceneSeen detections:

1. Catalogue retrieval with ground truth (leave one image out). For each product with at least
   two photos, one photo is turned into a query exactly like a video detection: its OWLv2
   garment box + padding (the photos where no garment box was found are not used as queries).
   That photo is removed from the index; the other photos of the product must bring it back.
   Product-type filtering, colour and ranking are the production code (MultiBrandCatalogue.rank).
   Reports top-1 / top-3 product accuracy per variant. Limitation: queries are studio photos of
   the same shoot, so absolute numbers are higher than on video; use it to compare variants.

2. Real detections from analysed videos (no ground truth of which product is on screen):
   colour agreement of the top 3, how much the lists change, and contact sheets
   (detected crop | baseline top 3 | new top 3) for visual review.

    python scripts/product_matching_eval.py                       # all variants, writes reports/matching/
    python scripts/product_matching_eval.py --queries 400 --sheets 24

Needs both indexes: embeddings.npz (full, the original) and embeddings_garment.npz
(python -m sceneseen match-catalogue --build-index --mode all).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sceneseen.commercial.attributes import dominant_color  # noqa: E402
from sceneseen.commercial.garment_crop import crop_fraction, letterbox  # noqa: E402
from sceneseen.commercial.matching import (CatalogueIndex, ClipEncoder, MultiBrandCatalogue,  # noqa: E402
                                           color_confidence)
from sceneseen.commercial.product_taxonomy import DetectionQuery, load_taxonomy  # noqa: E402
from sceneseen.config import catalogue_dirs, load_config  # noqa: E402

# name → (catalogue images, letterbox query + catalogue crops, ranking options)
VARIANTS = {
    "A_full_photos": ("full", False, {"improved": False, "aggregate": "max"}),
    "B_garment": ("garment", False, {"improved": False, "aggregate": "max"}),
    "B_garment_letterbox": ("garment", True, {"improved": False, "aggregate": "max"}),
    "C_garment_ranking": ("garment", False, {"improved": True, "aggregate": "max"}),
    "C_garment_ranking_top2": ("garment", False, {"improved": True, "aggregate": "top2"}),
    "C_garment_letterbox_ranking": ("garment", True, {"improved": True, "aggregate": "max"}),
}


def query_for_type(tax, ptype: str, color: str | None) -> DetectionQuery:
    """What a detection of this garment would be compared with (the production detection mapping)."""
    for det, spec in tax.detections.items():
        if ptype in spec.get("strict", []):
            return DetectionQuery(ptype, det, tuple(spec["strict"]), tuple(spec.get("broad", [])), color)
    umb = tax.types[ptype]["umbrella"]
    return DetectionQuery(ptype, None, (ptype,), tuple(t for t in tax.order if t != ptype and tax.types[t]["umbrella"] == umb),
                          color)


class Masked:
    """A MultiBrandCatalogue view of one index with one row hidden (the query photo)."""

    def __init__(self, idx: CatalogueIndex, emb: np.ndarray):
        self.idx, self.emb = idx, emb
        self.infos = idx.infos()

    def catalogue(self, drop: int) -> MultiBrandCatalogue:
        v = CatalogueIndex(self.idx.brand, self.idx.brand_dir, self.idx.mode)
        v.meta = self.idx.meta
        v.items = self.idx.items[:drop] + self.idx.items[drop + 1:]
        v.embeddings = np.delete(self.emb, drop, axis=0)
        v._infos = self.infos[:drop] + self.infos[drop + 1:]
        mb = MultiBrandCatalogue([], image_mode=self.idx.mode)
        mb.indices = {self.idx.brand_dir.name: v}
        return mb


def rank(mb, vec, q, opts, conf):
    if opts["improved"]:
        return mb.rank(vec, q, 3, color_confidence=conf, aggregate=opts["aggregate"])["candidates"]
    # the ranking before this change: full colour weight whenever a colour was detected, name colours only
    return mb.rank(vec, q, 3, color_confidence=1.0 if q.color else 0.0, aggregate="max",
                   image_colors=False)["candidates"]


def part1(cc, brand_dir: Path, n_queries: int, enc: ClipEncoder, seed: int = 0) -> dict:
    tax = load_taxonomy()
    full = CatalogueIndex(brand_dir.name, brand_dir, "full")
    garm = CatalogueIndex(brand_dir.name, brand_dir, "garment")
    assert full.load() and garm.load(), "build both indexes first (--build-index --mode all)"
    # the same photo set in both indexes, in the same order
    pos = {it.image_rel: i for i, it in enumerate(full.items)}
    keep = [i for i, it in enumerate(garm.items) if it.image_rel in pos]
    garm.items = [garm.items[i] for i in keep]
    garm.embeddings = garm.embeddings[keep]
    garm._infos = None
    order = [pos[it.image_rel] for it in garm.items]
    full.items = [full.items[i] for i in order]
    full.embeddings = full.embeddings[order]
    full._infos = None
    infos = garm.infos()
    by_product: dict[str, list[int]] = {}
    for r, (it, inf) in enumerate(zip(garm.items, infos)):
        if inf.product_type:
            by_product.setdefault(it.product_id, []).append(r)
    rng = np.random.default_rng(seed)
    queries = []
    for pid, rows in sorted(by_product.items()):
        usable = [r for r in rows if garm.items[r].crop_method in ("detector", "union")]
        if len(rows) >= 2 and usable:
            queries.append(int(rng.choice(usable)))
    rng.shuffle(queries)
    queries = queries[:n_queries]
    print(f"part 1: {len(queries)} query photos ({len(by_product)} typed products)")

    # query crops (as a video detection would be cropped) and their embeddings
    crops = []
    for r in queries:
        it = garm.items[r]
        with Image.open(brand_dir / it.image_path) as im:
            crops.append(crop_fraction(im.convert("RGB"), it.crop_box))
    q_centre = enc.embed(crops)
    q_letter = enc.embed([letterbox(c) for c in crops])
    colours = [dominant_color(np.asarray(c), [0, 0, 1, 1]) for c in crops]
    confs = [color_confidence(c, col) for c, col in zip(crops, colours)]

    # the saved garment crops embedded both ways (whatever the index itself was built with)
    print("  embedding the garment crops with and without letterbox …")
    ims = []
    for it in garm.items:
        with Image.open(brand_dir / it.crop_path) as im:
            ims.append(im.convert("RGB"))
    garm_emb = {False: enc.embed(ims), True: enc.embed([letterbox(im) for im in ims])}

    res = {}
    for name, (mode, lb, opts) in VARIANTS.items():
        idx = full if mode == "full" else garm
        emb = full.embeddings if mode == "full" else garm_emb[lb]
        view = Masked(idx, emb)
        hits1 = hits3 = 0
        by_umb: dict[str, list[int]] = {}
        for k, r in enumerate(queries):
            it, inf = garm.items[r], infos[r]
            q = query_for_type(tax, inf.product_type, colours[k])
            vec = (q_letter if lb else q_centre)[k]
            cands = rank(view.catalogue(r), vec, q, opts, confs[k])
            ids = [c.product_id for c in cands]
            h1, h3 = bool(ids[:1] == [it.product_id]), it.product_id in ids
            hits1 += h1
            hits3 += h3
            by_umb.setdefault(inf.umbrella, [0, 0, 0])
            by_umb[inf.umbrella][0] += 1
            by_umb[inf.umbrella][1] += h1
            by_umb[inf.umbrella][2] += h3
        n = len(queries)
        res[name] = {"queries": n, "top1": round(hits1 / n, 4), "top3": round(hits3 / n, 4),
                     "by_umbrella": {u: {"n": v[0], "top1": round(v[1] / v[0], 3), "top3": round(v[2] / v[0], 3)}
                                     for u, v in sorted(by_umb.items())}}
        print(f"  {name:<26} top-1 {hits1 / n:6.1%}   top-3 {hits3 / n:6.1%}")
    res["_queries_with_colour"] = sum(1 for c in colours if c)
    return res


def part2(cfg, enc: ClipEncoder, sheets: int, out_dir: Path) -> dict:
    """Real detections from every analysed video in data/cache."""
    from sceneseen.commercial.frames import frame_name
    from sceneseen.commercial.matching import _crop

    cc = cfg.catalogue
    tax = load_taxonomy()
    roots = catalogue_dirs(cc)
    base = MultiBrandCatalogue(roots, "full")
    base.discover_and_load()
    new = MultiBrandCatalogue(roots, "garment")
    new.discover_and_load()
    objs = []
    for cdir in sorted((ROOT / "data" / "cache").iterdir()):
        rec_f = sorted((cdir / "commercial").glob("detections_*.json")) if (cdir / "commercial").exists() else []
        frames = cdir / "commercial" / f"frames_{cfg.commercial.frame_long_side}"
        if not rec_f or not frames.exists():
            continue
        # candidates come from the commercial record the server builds
        try:
            from server.app import _commercial
            rec = _commercial(cdir.name, allow_inference=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {cdir.name}: no commercial record ({e})")
            continue
        for sc in rec.get("scenes", []):
            for c in sc["candidates"]:
                if not c.get("displayed"):
                    continue
                q = tax.detection_query(c["label"], c["type_id"], c.get("color"))
                if not q.strict_types:
                    continue
                f = frames / frame_name(c["best"]["shot_id"], c["best"]["position"])
                if f.exists():
                    objs.append((cdir.name, c, q, _crop(f, c["best"]["box"], cc.crop_pad)))
    print(f"part 2: {len(objs)} clothing / accessory detections with catalogue-compatible types")
    # one colour rule for both lists: the product name's colour, else the garment-crop colour of the shown photo
    crop_col = {it.image_rel: (it.crop_color if it.crop_method in ("detector", "union") else None)
                for idx in new.indices.values() if idx.mode == "garment" for it in idx.items}

    def colour_rel(q, m):
        cols = tuple(m.colors) or ((crop_col.get(m.best_image_rel),) if crop_col.get(m.best_image_rel) else ())
        if not cols:
            return "unknown"
        from sceneseen.commercial.product_taxonomy import ProductInfo
        return tax.color_relation(q.color, ProductInfo(None, None, cols))

    crops = [o[3] for o in objs]
    v_c = enc.embed(crops)
    v_l = enc.embed([letterbox(c) for c in crops])
    stats = {"objects": len(objs), "with_colour": 0, "baseline": {"same_or_near": 0, "different": 0, "unknown": 0},
             "new": {"same_or_near": 0, "different": 0, "unknown": 0}, "first_changed": 0, "top3_overlap": 0}
    rows = []
    for k, (vid, c, q, crop) in enumerate(objs):
        conf = color_confidence(crop, q.color)
        a = rank(base, v_c[k], q, {"improved": False}, conf)
        vecs = {m: (v_l if m[1] else v_c)[k] for m in new.query_modes()}
        b = new.rank(vecs, q, 3, color_confidence=conf, aggregate=cc.aggregate)["candidates"]
        if q.color:
            stats["with_colour"] += 1
            for key, lst in (("baseline", a), ("new", b)):
                for m in lst:
                    rel = colour_rel(q, m)
                    stats[key]["same_or_near" if rel in ("same", "near") else rel] += 1
        stats["first_changed"] += bool(a and b and a[0].product_id != b[0].product_id)
        stats["top3_overlap"] += len({m.product_id for m in a} & {m.product_id for m in b})
        rows.append((vid, c, q, crop, a, b))
    stats["top3_overlap_mean"] = round(stats["top3_overlap"] / max(1, len(objs)), 2)
    if sheets:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_sheets(rows[:sheets], new, out_dir)
    return stats


def write_sheets(rows, new: MultiBrandCatalogue, out_dir: Path, per: int = 6) -> None:
    T = 170
    brand_dirs = {s: i.brand_dir for s, i in new.indices.items()}
    for s in range(0, len(rows), per):
        part = rows[s:s + per]
        sheet = Image.new("RGB", (T * 7 + 20, len(part) * (T + 34)), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, (vid, c, q, crop, a, b) in enumerate(part):
            y = i * (T + 34)
            tiles = [(crop, f"{c['label']}")]
            for m in a:
                with Image.open(brand_dirs[m.brand_slug] / "images" / m.best_image_rel) as im:
                    tiles.append((im.convert("RGB"), f"A {m.similarity:.2f} {m.name}"))
            while len(tiles) < 4:
                tiles.append((None, "A —"))
            for m in b:
                with Image.open(brand_dirs[m.brand_slug] / m.best_crop_rel) as im:
                    tiles.append((im.convert("RGB"), f"C {m.similarity:.2f} {m.name}"))
            for j, (im, label) in enumerate(tiles):
                x = j * T + (20 if j >= 4 else 0)
                if im is not None:
                    im = im.copy()
                    im.thumbnail((T - 4, T - 4))
                    sheet.paste(im, (x + 2, y + 2))
                d.text((x + 2, y + T + 2), label[:28], fill=(255, 255, 255))
                d.text((x + 2, y + T + 15), label[28:56], fill=(200, 200, 200))
        sheet.save(out_dir / f"detections_{s // per:02d}.jpg", quality=85)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--brand", default="in_your_shoe")
    ap.add_argument("--queries", type=int, default=600)
    ap.add_argument("--sheets", type=int, default=30, help="detections drawn on contact sheets (0 = none)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "matching"))
    args = ap.parse_args(argv)
    cfg = load_config()
    cc = cfg.catalogue
    brand_dir = next((r / args.brand for r in catalogue_dirs(cc) if (r / args.brand / "catalogue.jsonl").exists()), None)
    if brand_dir is None:
        print(f"brand folder '{args.brand}' not found in {catalogue_dirs(cc)}")
        return 1
    enc = ClipEncoder(cc.model, cc.pretrained, cc.device)
    report = {"date": dt.datetime.now().isoformat(timespec="seconds"), "brand": args.brand,
              "model": f"{cc.model}/{cc.pretrained}", "taxonomy": load_taxonomy().version,
              "catalogue_retrieval": part1(cc, brand_dir, args.queries, enc)}
    sheets_dir = ROOT / "data" / "cache" / "matching_eval"
    report["real_detections"] = part2(cfg, enc, args.sheets, sheets_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"eval_{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    f.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["real_detections"], indent=1))
    print(f"report: {f}\ncontact sheets (not in Git: catalogue photos): {sheets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
