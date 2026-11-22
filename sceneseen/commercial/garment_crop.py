"""Garment-level crops of catalogue images (Phase 2B).

Catalogue photos show a model wearing a whole outfit; the product is one garment in it. Embedding
the full photo makes "Black Wide Leg Pants" look like the white shirt worn with them. Here each
catalogue image is cropped to the garment being sold:

    product type (from product_taxonomy.json, e.g. "jeans")
        │  its crop prompts ("jeans")                 sets: best box per part, unioned
        ▼
    OWLv2 (the detector Phase 2A already uses) on the catalogue image → best box of those prompts
        ▼
    box + the same padding as video detections → letterboxed to a square → OpenCLIP

Nothing is cropped blindly: an image with no confident box for its product's prompts keeps the
full image (method "full", with the reason), and so does a product whose type is unknown.

Raw detections are cached per image in <brand>/garment_boxes.json (keyed by file size + mtime),
so changing the selection rules or the padding never re-runs the detector.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DETECTOR_KEY = "owlv2"
BOXES_FORMAT = 2
STORE_THRESHOLD = 0.08        # raw detections kept in the cache (selection uses min_score below)


@dataclass
class GarmentCrop:
    box: list[float]            # chosen garment box, fractions of the image, BEFORE padding
    method: str                 # detector | union | full
    score: float | None = None  # detector score of the chosen box (lowest part for a union)
    prompt: str = ""
    reason: str = ""            # why the full image was kept

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = [round(float(v), 4) for v in self.box]
        if self.score is not None:
            d["score"] = round(float(self.score), 4)
        return d


# ---------------------------------------------------------------- image helpers (shared with video crops)

def pad_box(box: list[float], pad: float) -> list[float]:
    """Box grown by `pad` × its size on every side, clipped to the image (fractions)."""
    x1, y1, x2, y2 = box
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    return [max(0.0, x1 - px), max(0.0, y1 - py), min(1.0, x2 + px), min(1.0, y2 + py)]


def crop_fraction(im: Image.Image, box: list[float]) -> Image.Image:
    w, h = im.size
    x1, y1, x2, y2 = box
    return im.crop((int(x1 * w), int(y1 * h), max(int(x1 * w) + 1, int(x2 * w)), max(int(y1 * h) + 1, int(y2 * h))))


def letterbox(im: Image.Image, fill: tuple[int, int, int] = (124, 116, 104)) -> Image.Image:
    """Pad to a square (fill ≈ CLIP's mean colour) so CLIP's centre crop cuts nothing off a tall or
    wide garment: trousers keep their legs, a cap keeps its brim."""
    im = im.convert("RGB")
    w, h = im.size
    if w == h:
        return im
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


# ---------------------------------------------------------------- selection

def crop_spec(taxonomy, product_type: str | None) -> list[list[str]] | None:
    """[[prompts of part 1], [prompts of part 2]…] for a product type, or None (no crop)."""
    if not product_type or product_type not in taxonomy.types:
        return None
    spec = taxonomy.types[product_type].get("crop")
    if not spec:
        return None
    return [list(p) for p in spec["parts"]] if "parts" in spec else [list(spec["prompts"])]


def crop_options(taxonomy, product_type: str | None) -> dict:
    """Extra selection rules of a product type: {"outer": [prompts], "min_area": fraction}."""
    spec = (taxonomy.types.get(product_type) or {}).get("crop") or {} if product_type else {}
    return {"outer": list(spec.get("outer", [])), "min_area": float(spec.get("min_area", 0.0))}


def all_prompts(taxonomy) -> list[str]:
    out: list[str] = []
    for t in taxonomy.order:
        for p in [p for part in crop_spec(taxonomy, t) or [] for p in part] + crop_options(taxonomy, t)["outer"]:
            if p not in out:
                out.append(p)
    return out


def select_crop(dets: list[dict], parts: list[list[str]] | None, min_score: float,
                outer: list[str] | tuple = (), min_area: float = 0.0) -> GarmentCrop:
    """Pick the garment box from cached detections [{prompt, score, box}].

    One part: the highest-scoring box of its prompts, ignoring boxes smaller than min_area of the
    photo (a collar is not the polo). Several parts (sets, pyjamas): the best box of each part that
    has one, unioned; the first part (the top) is required.
    outer: layering. A shirt worn open over a tee is found by OWLv2 as "shirt jacket", while its
    "shirt" box is the tee underneath. If a higher-scoring box of an `outer` prompt contains the
    chosen box, the product is that outer layer and its box is used.
    """
    full = [0.0, 0.0, 1.0, 1.0]
    if not parts:
        return GarmentCrop(full, "full", reason="product type unknown or has no crop prompts")
    best_per_part = []
    for part in parts:
        cands = [d for d in dets if d["prompt"] in part and d["score"] >= min_score and _area(d["box"]) >= min_area]
        best_per_part.append(max(cands, key=lambda d: (d["score"], _area(d["box"]))) if cands else None)
    if best_per_part[0] is not None and outer:
        b = best_per_part[0]
        covers = [d for d in dets if d["prompt"] in outer and d["score"] > b["score"]
                  and _contains(d["box"], b["box"]) and _area(d["box"]) > 1.2 * _area(b["box"])]
        if covers:
            best_per_part[0] = max(covers, key=lambda d: d["score"])
    if best_per_part[0] is None:
        top = max((d["score"] for d in dets if d["prompt"] in parts[0]), default=None)
        return GarmentCrop(full, "full", reason=f"no box for {'/'.join(parts[0])} ≥ {min_score}"
                                                + (f" (best {top:.2f})" if top is not None else ""))
    found = [d for d in best_per_part if d is not None]
    if len(found) == 1:
        d = found[0]
        return GarmentCrop(list(d["box"]), "detector", d["score"], d["prompt"])
    bs = [d["box"] for d in found]
    box = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
    return GarmentCrop(box, "union", min(d["score"] for d in found), "+".join(d["prompt"] for d in found))


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contains(outer: list[float], inner: list[float], frac: float = 0.85) -> bool:
    """At least `frac` of the inner box lies inside the outer box."""
    ix = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    iy = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    a = _area(inner)
    return a > 0 and ix * iy / a >= frac


# ---------------------------------------------------------------- detection

def detect_per_prompt(det, images: list[Image.Image], prompts: list[str], per_prompt: int = 3) -> list[list[dict]]:
    """For every image and EVERY prompt, its best boxes scored by that prompt alone.

    The usual OWLv2 post-processing gives each box only its single best-scoring prompt, so with
    50 garment prompts a cardigan box comes back labelled "knit top" and the cardigan prompt
    never wins. Product crops need "where is the cardigan?", hence one ranking per prompt.
    """
    import torch

    det.load()
    queries = [det.spec.prompt_template.format(p) for p in prompts]

    def run():
        with torch.no_grad():
            inp = det._proc(text=[queries] * len(images), images=images, return_tensors="pt").to(det.device)
            out = det._model(**inp)
        return torch.sigmoid(out.logits).float().cpu().numpy(), out.pred_boxes.float().cpu().numpy()

    try:
        scores, boxes = run()
    except Exception as e:
        if det.device == "cpu":
            raise
        det._to_cpu(e)
        scores, boxes = run()
    results = []
    for im, sc, bx in zip(images, scores, boxes):
        w, h = im.size
        s = max(w, h)                                   # OWLv2 pads to a square: boxes are fractions of it
        cx, cy, bw, bh = bx[:, 0] * s / w, bx[:, 1] * s / h, bx[:, 2] * s / w, bx[:, 3] * s / h
        xyxy = np.clip(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), 0.0, 1.0)
        dets = []
        for qi, p in enumerate(prompts):
            order = np.argsort(-sc[:, qi])
            kept: list[list[float]] = []
            for n in order:
                if sc[n, qi] < STORE_THRESHOLD or len(kept) >= per_prompt:
                    break
                b = xyxy[n].tolist()
                if b[2] - b[0] < 0.01 or b[3] - b[1] < 0.01 or any(_iou(b, k) > 0.5 for k in kept):
                    continue
                kept.append(b)
                dets.append({"prompt": p, "score": round(float(sc[n, qi]), 4), "box": [round(v, 4) for v in b]})
        results.append(dets)
    return results


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- detection cache

class GarmentBoxes:
    """<brand>/garment_boxes.json: raw OWLv2 detections of every catalogue image for every crop prompt."""

    def __init__(self, brand_dir: Path):
        self.path = Path(brand_dir) / "garment_boxes.json"
        self.data: dict = {"format": BOXES_FORMAT, "detector": "", "prompts": [], "images": {}}
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if d.get("format") == BOXES_FORMAT:
                    self.data = d
            except Exception as e:
                log.warning("unreadable %s (%s); detections will be recomputed", self.path, e)

    def get(self, rel: str, size: int | None, mtime_ns: int | None, prompts: list[str]) -> list[dict] | None:
        rec = self.data["images"].get(rel)
        if rec is None or (rec.get("size"), rec.get("mtime_ns")) != (size, mtime_ns):
            return None
        if not set(prompts) <= set(self.data.get("prompts", [])):
            return None
        return rec["dets"]

    def missing(self, items, prompts: list[str]) -> list:
        return [it for it in items if self.get(it.image_rel, it.image_size, it.image_mtime_ns, prompts) is None]

    def detect(self, brand_dir: Path, items, prompts: list[str], device: str = "auto", batch_size: int = 8,
               progress: Callable[[int, int], None] | None = None, save_every: int = 200) -> int:
        """Run OWLv2 on every item without cached detections. Returns how many images were processed."""
        from .detector import MODELS, get_detector

        todo = self.missing(items, prompts)
        if not todo:
            return 0
        if set(prompts) != set(self.data.get("prompts", [])):
            if self.data["images"]:
                log.info("crop prompts changed: re-detecting every catalogue image")
            self.data = {"format": BOXES_FORMAT, "detector": MODELS[DETECTOR_KEY].model_id, "prompts": list(prompts),
                         "images": {}}
            todo = list(items)
        det = get_detector(DETECTOR_KEY, device, STORE_THRESHOLD, batch_size)
        done = 0
        for i in range(0, len(todo), batch_size):
            chunk, ims = [], []
            for it in todo[i:i + batch_size]:
                try:
                    with Image.open(Path(brand_dir) / it.image_path) as im:
                        ims.append(im.convert("RGB"))
                    chunk.append(it)
                except Exception as e:
                    log.warning("cannot read %s for garment detection: %s", it.image_path, e)
            if ims:
                for it, dets in zip(chunk, detect_per_prompt(det, ims, prompts)):
                    self.data["images"][it.image_rel] = {"size": it.image_size, "mtime_ns": it.image_mtime_ns,
                                                         "dets": dets}
            done += len(chunk)
            if progress:
                progress(done, len(todo))
            if done % save_every < batch_size:
                self.save()
        self.save()
        return done

    def detected_images(self) -> int:
        return len(self.data["images"])

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------- inspection

def crop_sheet(index, n: int = 48, seed: int = 0, out: Path | None = None) -> Path:
    """Contact sheet of n sampled catalogue images with their garment box drawn (red = garment box,
    blue = full photo kept), to check by eye that the right garment was isolated."""
    from PIL import ImageDraw

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(index.items), size=min(n, len(index.items)), replace=False)
    tw, th, cols = 260, 340, 8
    sheet = Image.new("RGB", (cols * tw, ((len(rows) + cols - 1) // cols) * th), (40, 40, 40))
    for k, r in enumerate(rows):
        it = index.items[r]
        with Image.open(index.brand_dir / it.image_path) as im:
            im = im.convert("RGB")
        im.thumbnail((tw, th - 30))
        d = ImageDraw.Draw(im)
        w, h = im.size
        b = it.crop_box or [0, 0, 1, 1]
        full = it.crop_method in ("", "full")
        d.rectangle([b[0] * w, b[1] * h, b[2] * w - 1, b[3] * h - 1], outline=(40, 90, 255) if full else (230, 30, 30),
                    width=4)
        tile = Image.new("RGB", (tw, th), (20, 20, 20))
        tile.paste(im, (0, 30))
        td = ImageDraw.Draw(tile)
        td.text((3, 2), f"{it.crop_method} {it.crop_prompt}"[:44], fill=(255, 220, 0))
        td.text((3, 15), it.name[:44], fill=(255, 255, 255))
        sheet.paste(tile, ((k % cols) * tw, (k // cols) * th))
    out = out or index.brand_dir / "crop_inspection.jpg"
    sheet.save(out, quality=85)
    return out
