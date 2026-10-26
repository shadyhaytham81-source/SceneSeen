"""Product catalogue matching for SceneSeen (Phase 2B).

Uses OpenCLIP image embeddings to match detected commercial objects (from OWLv2)
against clothing brand catalogues (starting with IN YOUR SHOE).

Detections and matching remain completely separate:
- OWLv2 detects commercial objects and bounding boxes.
- OpenCLIP embeddings + cosine similarity retrieve top candidate products.
- Candidates are ranked by visual similarity (not asserted as ground-truth exact matches).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


@dataclass
class CatalogueItem:
    brand: str
    product_id: str
    handle: str
    name: str
    category: str
    product_type: str
    price: float | None
    compare_at_price: float | None
    currency: str
    url: str
    image_path: str          # relative to repo root or absolute
    image_rel: str           # brand-relative path: e.g. in_your_shoe/<handle>/<filename>
    image_filename: str
    image_url: str = ""      # original CDN url if available

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProductMatch:
    product_id: str
    brand: str
    name: str
    category: str
    product_type: str
    price: float | None
    compare_at_price: float | None
    currency: str
    url: str
    similarity: float
    best_image_rel: str
    best_image_url: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["similarity"] = round(float(self.similarity), 4)
        return d


class CatalogueIndex:
    """Persistent index of catalogue image embeddings for one or more brands."""

    def __init__(self, brand: str, brand_dir: Path):
        self.brand = brand
        self.brand_dir = Path(brand_dir)
        self.embeddings_path = self.brand_dir / "embeddings.npz"
        self.manifest_path = self.brand_dir / "embeddings_manifest.json"
        self.items: list[CatalogueItem] = []
        self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)

    def is_built(self) -> bool:
        return self.embeddings_path.exists() and self.manifest_path.exists()

    def load(self) -> bool:
        """Load persistent embeddings and manifest from disk."""
        if not self.is_built():
            return False
        try:
            with np.load(self.embeddings_path) as npz:
                self.embeddings = npz["embeddings"].astype(np.float32)
            raw_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.items = [CatalogueItem(**item) for item in raw_manifest]
            if len(self.items) != len(self.embeddings):
                log.warning("Manifest count (%d) mismatch with embeddings (%d) in %s; rebuild needed",
                            len(self.items), len(self.embeddings), self.brand_dir)
                return False
            log.info("Loaded %d embeddings for brand '%s' from %s",
                     len(self.items), self.brand, self.embeddings_path)
            return True
        except Exception as e:
            log.warning("Failed loading catalogue embeddings from %s: %s", self.embeddings_path, e)
            return False

    def build_or_update(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str = "auto",
        batch_size: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Build or incrementally update image embeddings from catalogue.jsonl and image files."""
        from sceneseen.features import load_clip, resolve_device
        import torch

        cat_file = self.brand_dir / "catalogue.jsonl"
        images_dir = self.brand_dir / "images"
        if not cat_file.exists() or not images_dir.exists():
            raise FileNotFoundError(f"Missing catalogue.jsonl or images/ in {self.brand_dir}")

        # Check existing items to avoid reprocessing
        existing_map: dict[str, int] = {}
        if self.load():
            for idx, it in enumerate(self.items):
                existing_map[it.image_rel] = idx

        # Discover all valid images per product
        products = [json.loads(line) for line in cat_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

        to_embed_paths: list[Path] = []
        to_embed_items: list[CatalogueItem] = []
        kept_items: list[CatalogueItem] = []
        kept_indices: list[int] = []

        for p in products:
            handle = p.get("handle") or str(p.get("product_id"))
            brand_name = p.get("brand") or self.brand
            # Look in brand/handle or directly handle
            folder = images_dir / self.brand_dir.name / handle
            if not folder.exists():
                folder = images_dir / handle
            if not folder.exists():
                continue

            cdn_urls = p.get("image_urls") or []
            # Match image files
            img_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts],
                               key=lambda x: x.name)
            for idx, img_path in enumerate(img_files):
                rel_path = f"{self.brand_dir.name}/{handle}/{img_path.name}"
                cdn_url = cdn_urls[idx] if idx < len(cdn_urls) else (cdn_urls[0] if cdn_urls else "")
                item = CatalogueItem(
                    brand=brand_name,
                    product_id=str(p.get("product_id") or handle),
                    handle=handle,
                    name=p.get("name") or handle,
                    category=p.get("category") or p.get("product_type") or "",
                    product_type=p.get("product_type") or "",
                    price=float(p["price"]) if p.get("price") is not None else None,
                    compare_at_price=float(p["compare_at_price"]) if p.get("compare_at_price") is not None else None,
                    currency=p.get("currency") or "EGP",
                    url=p.get("url") or "",
                    image_path=str(img_path),
                    image_rel=rel_path,
                    image_filename=img_path.name,
                    image_url=cdn_url,
                )
                if rel_path in existing_map:
                    kept_items.append(item)
                    kept_indices.append(existing_map[rel_path])
                else:
                    to_embed_paths.append(img_path)
                    to_embed_items.append(item)

        log.info("Brand '%s': %d images already indexed, %d new images to embed",
                 self.brand, len(kept_items), len(to_embed_paths))

        if not to_embed_paths and self.items and len(kept_items) == len(self.items):
            return 0

        new_embeddings_list: list[np.ndarray] = []
        valid_new_items: list[CatalogueItem] = []
        if to_embed_paths:
            dev = resolve_device(device)
            model, preprocess = load_clip(model_name, pretrained, dev)
            total = len(to_embed_paths)

            for i in range(0, total, batch_size):
                batch_paths = to_embed_paths[i:i + batch_size]
                tensors = []

                for bi, pth in enumerate(batch_paths):
                    try:
                        with Image.open(pth) as im:
                            tensors.append(preprocess(im.convert("RGB")))
                        valid_new_items.append(to_embed_items[i + bi])
                    except Exception as e:
                        log.warning("Skipping corrupt or unreadable catalogue image %s: %s", pth, e)

                if tensors:
                    batch_tensor = torch.stack(tensors).to(dev)
                    with torch.no_grad():
                        embs = model.encode_image(batch_tensor).float()
                        embs = embs / embs.norm(dim=-1, keepdim=True)
                    new_embeddings_list.append(embs.cpu().numpy().astype(np.float32))

                if progress:
                    progress(min(total, i + batch_size), total)

        # Merge kept + new
        all_items: list[CatalogueItem] = []
        all_embs: list[np.ndarray] = []

        if kept_indices and len(self.embeddings) > 0:
            all_items.extend(kept_items)
            all_embs.append(self.embeddings[kept_indices])

        if new_embeddings_list:
            all_items.extend(valid_new_items)
            all_embs.append(np.vstack(new_embeddings_list))

        if all_embs:
            self.embeddings = np.vstack(all_embs).astype(np.float32)
            self.items = all_items

            # Save persistently
            self.brand_dir.mkdir(parents=True, exist_ok=True)
            tmp_npz = self.embeddings_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_npz, embeddings=self.embeddings)
            tmp_npz.replace(self.embeddings_path)

            tmp_manifest = self.manifest_path.with_suffix(".tmp.json")
            tmp_manifest.write_text(json.dumps([it.to_dict() for it in self.items], indent=1), encoding="utf-8")
            tmp_manifest.replace(self.manifest_path)

            log.info("Saved %d embeddings and manifest for brand '%s'", len(self.items), self.brand)

        return len(valid_new_items)


class MultiBrandCatalogue:
    """Manages catalogue indices across multiple brands."""

    def __init__(self, roots: list[Path] | None = None):
        if roots is None:
            # Default lookup paths: scraper/output and data/catalogues
            from sceneseen.config import ROOT
            roots = [ROOT / "scraper" / "output", ROOT / "data" / "catalogues"]
        self.roots = [Path(r) for r in roots]
        self.indices: dict[str, CatalogueIndex] = {}

    def discover_and_load(self) -> None:
        """Find and load all available brand catalogue indices without re-embedding."""
        for root in self.roots:
            if not root.exists():
                continue
            for brand_dir in root.iterdir():
                if brand_dir.is_dir() and (brand_dir / "catalogue.jsonl").exists():
                    brand_slug = brand_dir.name
                    if brand_slug not in self.indices:
                        idx = CatalogueIndex(brand=brand_slug, brand_dir=brand_dir)
                        if idx.is_built():
                            idx.load()
                        self.indices[brand_slug] = idx

    def total_products(self) -> int:
        return sum(len(set(it.product_id for it in idx.items)) for idx in self.indices.values())

    def total_images(self) -> int:
        return sum(len(idx.items) for idx in self.indices.values())

    def match_crop(
        self,
        crop_image: Image.Image,
        model,
        preprocess,
        device: str,
        top_k: int = 3,
        brand_filter: str | None = None,
    ) -> list[ProductMatch]:
        """Compute visual similarity against catalogue images and return top-k unique products."""
        import torch

        # Ensure index is loaded
        if not self.indices:
            self.discover_and_load()

        # Target indices
        active_indices = [idx for b, idx in self.indices.items()
                          if (brand_filter is None or b == brand_filter) and len(idx.items) > 0]
        if not active_indices:
            return []

        # Encode query crop
        crop_tensor = preprocess(crop_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            crop_emb = model.encode_image(crop_tensor).float()
            crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)
        crop_vec = crop_emb.cpu().numpy().squeeze(0)  # shape (512,)

        # Compute cosine similarity across all products and take best image per product
        best_per_product: dict[str, tuple[float, CatalogueItem]] = {}

        for idx in active_indices:
            if len(idx.embeddings) == 0:
                continue
            sims = np.dot(idx.embeddings, crop_vec)  # shape (N,)
            for i, score in enumerate(sims):
                item = idx.items[i]
                pid = f"{item.brand}:{item.product_id}"
                if pid not in best_per_product or score > best_per_product[pid][0]:
                    best_per_product[pid] = (float(score), item)

        # Sort products by similarity descending
        sorted_products = sorted(best_per_product.values(), key=lambda x: -x[0])[:top_k]

        matches: list[ProductMatch] = []
        for score, it in sorted_products:
            matches.append(ProductMatch(
                product_id=it.product_id,
                brand=it.brand,
                name=it.name,
                category=it.category,
                product_type=it.product_type,
                price=it.price,
                compare_at_price=it.compare_at_price,
                currency=it.currency,
                url=it.url,
                similarity=score,
                best_image_rel=it.image_rel,
                best_image_url=it.image_url,
            ))
        return matches


def match_video_candidates(
    candidates: list[dict],
    frames_dir: Path,
    multi_brand: MultiBrandCatalogue | None = None,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str = "auto",
    top_k: int = 3,
) -> dict[str, list[dict]]:
    """Match all displayed commercial candidates for a video against the catalogue.

    Returns:
        {candidate_key: [match1_dict, match2_dict, match3_dict]}
    """
    from sceneseen.commercial.frames import frame_name
    from sceneseen.features import load_clip, resolve_device

    if multi_brand is None:
        multi_brand = MultiBrandCatalogue()
        multi_brand.discover_and_load()

    # Check if any brand has embeddings
    if multi_brand.total_images() == 0:
        log.warning("No catalogue embeddings available for matching.")
        return {}

    dev = resolve_device(device)
    model, preprocess = load_clip(model_name, pretrained, dev)

    matches_by_key: dict[str, list[dict]] = {}
    valid_cands = []
    crops = []

    for cand in candidates:
        ckey = cand.get("key")
        best = cand.get("best")
        if not ckey or not best:
            continue

        shot_id = best.get("shot_id", 0)
        pos = best.get("position", 0.5)
        box = best.get("box", [0, 0, 1, 1])

        frame_file = frames_dir / frame_name(shot_id, pos)
        if not frame_file.exists():
            continue

        try:
            with Image.open(frame_file) as im:
                im = im.convert("RGB")
                w, h = im.size
                x1, y1, x2, y2 = box
                bw, bh = (x2 - x1) * w, (y2 - y1) * h
                pad_x = 0.08 * bw
                pad_y = 0.08 * bh
                crop_box = (
                    int(max(0, x1 * w - pad_x)),
                    int(max(0, y1 * h - pad_y)),
                    int(min(w, x2 * w + pad_x)),
                    int(min(h, y2 * h + pad_y)),
                )
                crop_img = im.crop(crop_box)
                crops.append(crop_img)
                valid_cands.append(cand)
        except Exception as e:
            log.warning("Failed loading/cropping frame for candidate %s: %s", ckey, e)
            continue

    if not crops:
        return {}

    # Batched encoding for all crops
    import torch

    batch_size = 32
    crop_embs = []
    for bi in range(0, len(crops), batch_size):
        chunk = crops[bi:bi + batch_size]
        tensors = torch.stack([preprocess(c) for c in chunk]).to(dev)
        with torch.no_grad():
            embs = model.encode_image(tensors).float()
            embs = embs / embs.norm(dim=-1, keepdim=True)
        crop_embs.append(embs.cpu().numpy().astype(np.float32))

    all_crop_vecs = np.vstack(crop_embs)  # shape (M, 512)

    # Match each crop vector against all loaded indices
    active_indices = [idx for idx in multi_brand.indices.values() if len(idx.items) > 0]
    if not active_indices:
        return {}

    for ci, cand in enumerate(valid_cands):
        ckey = cand["key"]
        crop_vec = all_crop_vecs[ci]
        best_per_product: dict[str, tuple[float, CatalogueItem]] = {}

        for idx in active_indices:
            sims = np.dot(idx.embeddings, crop_vec)
            for i, score in enumerate(sims):
                item = idx.items[i]
                pid = f"{item.brand}:{item.product_id}"
                if pid not in best_per_product or score > best_per_product[pid][0]:
                    best_per_product[pid] = (float(score), item)

        sorted_products = sorted(best_per_product.values(), key=lambda x: -x[0])[:top_k]
        matches_by_key[ckey] = [
            ProductMatch(
                product_id=it.product_id,
                brand=it.brand,
                name=it.name,
                category=it.category,
                product_type=it.product_type,
                price=it.price,
                compare_at_price=it.compare_at_price,
                currency=it.currency,
                url=it.url,
                similarity=score,
                best_image_rel=it.image_rel,
                best_image_url=it.image_url,
            ).to_dict()
            for score, it in sorted_products
        ]

    return matches_by_key


def get_or_compute_matches(
    cache_dir: Path,
    candidates: list[dict],
    frames_dir: Path,
    multi_brand: MultiBrandCatalogue | None = None,
    allow_compute: bool = True,
) -> dict[str, list[dict]]:
    """Retrieve cached product matches, or compute and cache them."""
    matches_file = cache_dir / "commercial" / "matches.json"
    cached_matches: dict[str, list[dict]] = {}

    if matches_file.exists():
        try:
            cached_matches = json.loads(matches_file.read_text(encoding="utf-8"))
            displayed_keys = {c["key"] for c in candidates if c.get("displayed")}
            # If all displayed candidates already have cached matches, return
            if displayed_keys.issubset(cached_matches.keys()):
                return cached_matches
        except Exception as e:
            log.warning("Unreadable matches cache at %s: %s", matches_file, e)

    if not allow_compute:
        return cached_matches

    # Compute matches
    new_matches = match_video_candidates(candidates, frames_dir, multi_brand=multi_brand)
    cached_matches.update(new_matches)

    # Save cache
    try:
        matches_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = matches_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(cached_matches, indent=1), encoding="utf-8")
        tmp.replace(matches_file)
    except Exception as e:
        log.warning("Failed saving matches cache: %s", e)

    return cached_matches


def main():
    import argparse
    from sceneseen.config import ROOT

    parser = argparse.ArgumentParser(description="SceneSeen Product Catalogue Indexing & Matching (Phase 2B)")
    parser.add_argument("--build-index", action="store_true", help="Build or incrementally update catalogue embeddings")
    parser.add_argument("--brand", default="in_your_shoe", help="Brand folder name (default: in_your_shoe)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    root = ROOT / "scraper" / "output" / args.brand
    if not root.exists():
        root = ROOT / "data" / "catalogues" / args.brand

    if not root.exists():
        print(f"Error: brand catalogue folder not found at {root}")
        return 1

    idx = CatalogueIndex(brand=args.brand, brand_dir=root)
    if args.build_index or not idx.is_built():
        print(f"Building/updating index for brand: {args.brand} in {root}...")
        count = idx.build_or_update(batch_size=args.batch_size)
        print(f"Finished: {count} images embedded. Total indexed: {len(idx.items)}.")
    else:
        loaded = idx.load()
        print(f"Brand: {args.brand} | Index loaded: {loaded} | Total items: {len(idx.items)} | Embeddings: {idx.embeddings.shape}")
    return 0


if __name__ == "__main__":
    main()
