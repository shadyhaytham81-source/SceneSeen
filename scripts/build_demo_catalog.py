"""Build a DEMO catalogue from this project's own analysed videos (for trying the catalogue and the Identify Product workflow).

    python scripts/build_demo_catalog.py --config data/demo/demo.toml [--max-products 80]

Every product is a crop of an object SceneSeen detected in one of the analysed videos, saved under
a brand whose name ends in "(demo)". These are NOT real products and the images are film frames:
the catalogue lives only under data/ (git-ignored) and must never be committed or published.

The demo catalogue only gives the manual "Identify Product" workflow something to search in.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sceneseen.catalog import service as S          # noqa: E402
from sceneseen.catalog.db import Database           # noqa: E402
from sceneseen.catalog.images import ImageStore     # noqa: E402
from sceneseen.commercial import frames as F        # noqa: E402
from sceneseen.commercial import taxonomy as T      # noqa: E402
from sceneseen.config import catalog_paths, load_config  # noqa: E402


def select_occurrences(c: dict, limit: int) -> list[dict]:
    """The object's most confident detections, one per shot."""
    seen, out = set(), []
    for d in sorted(c.get("debug", {}).get("detections", []), key=lambda d: -d["score"]):
        if d["shot_id"] not in seen:
            seen.add(d["shot_id"])
            out.append(d)
    return out[:limit]


def crop_rgb(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return frame[int(max(0, (y1 - py) * h)):int(min(h, (y2 + py) * h)), int(max(0, (x1 - px) * w)):int(min(w, (x2 + px) * w))].copy()


BRANDS = {"fashion": "Nile Wear (demo)", "accessories": "Cairo Accessories (demo)", "electronics": "Delta Tech (demo)",
          "automotive": "Sahara Motors (demo)", "food_beverage": "Fayoum Foods (demo)", "furniture": "بيت الديكور (demo)",
          "beauty": "Lotus Beauty (demo)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-products", type=int, default=80)
    ap.add_argument("--per-video", type=int, default=12)
    a = ap.parse_args()
    cfg = load_config(a.config)
    url, img_dir = catalog_paths(cfg)
    if "demo" not in url:
        sys.exit("refusing to fill a catalogue that is not a demo database (use --config data/demo/demo.toml)")
    Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    db, store = Database(url), ImageStore(img_dir)
    brands = {b["name"]: b["id"] for b in S.list_brands(db, include_archived=True)}
    made = 0
    for vdir in sorted(Path(cfg.paths.cache_dir).iterdir()):
        cdir = vdir / "commercial"
        fdir = cdir / f"frames_{cfg.commercial.frame_long_side}"
        up = vdir / "upload.json"
        if not fdir.exists() or not up.exists() or made >= a.max_products:
            continue
        from sceneseen.pipeline import load_stage_outputs, unique_shots_for, VideoCache
        from sceneseen.media import VideoInfo, relocated
        from sceneseen.commercial import analysis
        from sceneseen import grouping
        from sceneseen.corrections import CorrectionStore
        from sceneseen.shots import Shot

        cache = VideoCache(cfg.paths.cache_dir, vdir.name)
        result, info_d = cache.read_json("result.json"), cache.read_json("info.json")
        if not result or not info_d:
            continue
        info_d["path"] = str(relocated(json.loads(up.read_text(encoding="utf-8"))["path"], cfg.paths.uploads_dir))
        info = VideoInfo(**info_d)
        stage = load_stage_outputs(cfg, info)
        scenes = result["scenes"]
        corr = CorrectionStore(cache.dir).load()
        if corr is not None:
            scenes = grouping.boundaries_to_scenes(corr["boundaries"], result["duration"], [Shot(**s) for s in result["debug"]["shots"]])
        unique = unique_shots_for(cfg, info, stage, scenes)
        com = analysis.analyze(stage["cache"].dir, info, stage["shots"], scenes, unique, cfg.commercial,
                               clip=stage["features"]["clip"], clip_model=(cfg.features.model, cfg.features.pretrained),
                               cache_root=cfg.paths.cache_dir, allow_inference=False)
        if com["status"] not in ("ready", "partial"):
            continue
        cands = [(sc["scene_id"], c) for sc in com["scenes"] for c in sc["candidates"] if c["displayed"]]
        cands.sort(key=lambda sc: -sc[1]["detection_confidence"])
        n = 0
        for scene_id, c in cands:
            if n >= a.per_video or made >= a.max_products:
                break
            crops = []
            for d in select_occurrences(c, 3):
                fp = fdir / F.frame_name(d["shot_id"], 0.5)
                if fp.exists():
                    crop = crop_rgb(F.load_rgb(fp), d["box"], 0.08)
                    if min(crop.shape[:2]) >= 140:
                        crops.append(crop)
            if not crops:
                continue
            obj = T.OBJECTS[c["type_id"]]
            bname = BRANDS.get(obj.category) or f"{T.CATEGORIES[obj.category].name} House (demo)"
            if bname not in brands:
                brands[bname] = S.create_brand(db, bname, description="Demo brand. Not a real company.")["id"]
            try:
                p = S.create_product(db, brands[bname], name=f"{obj.label} {vdir.name[:4].upper()}-S{scene_id:02d} (demo)",
                                     category=obj.category, object_type=obj.id, sku=f"DEMO-{c['key'][:8].upper()}",
                                     description="Demo product cut from an analysed video frame. Not a real product.",
                                     metadata={"demo": True, "video_id": vdir.name, "scene_id": scene_id, "candidate_key": c["key"]})
            except S.CatalogError:
                continue
            for i, crop in enumerate(crops):
                buf = io.BytesIO()
                Image.fromarray(crop).save(buf, "JPEG", quality=92)
                try:
                    S.add_image(db, store, p["id"], buf.getvalue(), "front" if i == 0 else "lifestyle", f"demo_{c['key']}_{i}.jpg")
                except S.CatalogError:
                    pass
            n += 1
            made += 1
        print(f"{vdir.name}: {n} demo products")
    print(json.dumps(S.stats(db)))


if __name__ == "__main__":
    main()
