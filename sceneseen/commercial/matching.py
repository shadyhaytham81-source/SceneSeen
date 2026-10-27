"""Product catalogue matching for SceneSeen (Phase 2B).

    OWLv2 detection ("Black jacket")
        │  product_taxonomy.py: normalise the label → product types + colour
        ▼
    catalogue filtered to COMPATIBLE product types (strict first, broader only if too few)
        │  OpenCLIP embedding of the detected crop (same model + preprocessing as the catalogue)
        ▼
    cosine similarity → best image per product → small colour adjustment → top 3 products

Responsibilities stay separate: OWLv2 finds objects; the taxonomy decides which products are
comparable; embeddings rank them visually. A similarity score is NOT a probability that the
product is the one on screen, and an object type with no compatible catalogue products gets
no candidates rather than unrelated ones.

Dataset layout (outside Git; folders from [catalogue] dirs or $SCENESEEN_CATALOGUE_DIR):

    <catalogue dir>/<brand_slug>/
        catalogue.jsonl              product records written by scraper/
        images/<brand_slug>/<handle>/<file>   (images/<handle>/<file> also accepted)
        embeddings.npz               float32 [N, D], L2-normalised, row i ↔ manifest item i
        embeddings_manifest.json     one record per row: brand, product id, name, image path, source URL, …
        embeddings_meta.json         model, pretrained weights, preprocessing fingerprint, dimension
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .product_taxonomy import DetectionQuery, ProductInfo, load_taxonomy

log = logging.getLogger(__name__)

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
INDEX_FORMAT = 2
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
BUILD_HINT = "python -m sceneseen match-catalogue --build-index"


# ---------------------------------------------------------------- records

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
    image_path: str          # relative to the brand folder (older indexes: absolute; not relied on)
    image_rel: str           # relative to <brand>/images/, e.g. in_your_shoe/<handle>/<file>
    image_filename: str
    image_url: str = ""      # source CDN URL when known
    brand_slug: str = ""     # brand folder name
    image_size: int | None = None        # file size and mtime when embedded: detects changed images
    image_mtime_ns: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CatalogueItem":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


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
    similarity: float                     # cosine similarity of the best image (not a probability)
    best_image_rel: str
    best_image_url: str = ""
    brand_slug: str = ""
    rank_score: float | None = None       # similarity + colour adjustment (what the list is sorted by)
    normalized_type: str | None = None
    umbrella: str | None = None
    colors: list = field(default_factory=list)
    color_relation: str = "unknown"       # same | near | different | unknown
    match_tier: str = "strict"            # strict | broad (fallback category)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["similarity"] = round(float(self.similarity), 4)
        if self.rank_score is not None:
            d["rank_score"] = round(float(self.rank_score), 4)
        return d


# ---------------------------------------------------------------- embedding model

class ClipEncoder:
    """OpenCLIP image encoder used for BOTH catalogue images and detected-object crops."""

    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED, device: str = "auto"):
        self.model_name, self.pretrained, self.device_pref = model_name, pretrained, device
        self._model = self._pre = self.device = None

    def load(self):
        if self._model is None:
            from sceneseen.features import load_clip, resolve_device

            self.device = resolve_device(self.device_pref)
            self._model, self._pre = load_clip(self.model_name, self.pretrained, self.device)
        return self

    @property
    def preprocess_fingerprint(self) -> str:
        self.load()
        return hashlib.sha1(repr(self._pre).encode()).hexdigest()[:12]

    def meta(self) -> dict:
        return {"model": self.model_name, "pretrained": self.pretrained, "preprocess": self.preprocess_fingerprint}

    def embed(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        import torch

        self.load()
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self._pre(im.convert("RGB")) for im in images[i:i + batch_size]]).to(self.device)
            with torch.no_grad():
                e = self._model.encode_image(batch).float()
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 0), np.float32)


# ---------------------------------------------------------------- one brand

class CatalogueIndex:
    """Persistent embeddings of one brand's catalogue images (built once, then loaded in milliseconds)."""

    def __init__(self, brand: str, brand_dir: Path):
        self.brand = brand
        self.brand_dir = Path(brand_dir)
        self.embeddings_path = self.brand_dir / "embeddings.npz"
        self.manifest_path = self.brand_dir / "embeddings_manifest.json"
        self.meta_path = self.brand_dir / "embeddings_meta.json"
        self.items: list[CatalogueItem] = []
        self.embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self.meta: dict = {}
        self.load_error: str | None = None
        self._infos: list[ProductInfo] | None = None

    # ---- files
    def has_catalogue(self) -> bool:
        return (self.brand_dir / "catalogue.jsonl").exists()

    def is_built(self) -> bool:
        return self.embeddings_path.exists() and self.manifest_path.exists()

    def products(self) -> list[dict]:
        cat = self.brand_dir / "catalogue.jsonl"
        out = []
        for line in cat.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    log.warning("skipping unreadable line in %s", cat)
        return out

    def load(self) -> bool:
        """Load embeddings + manifest (+ meta). False, with load_error set, if missing or inconsistent."""
        self.load_error = None
        self._infos = None
        if not self.is_built():
            self.load_error = "not built"
            return False
        try:
            with np.load(self.embeddings_path) as npz:
                self.embeddings = npz["embeddings"].astype(np.float32)
            self.items = [CatalogueItem.from_dict(x) for x in json.loads(self.manifest_path.read_text(encoding="utf-8"))]
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
        except Exception as e:
            self.load_error = f"unreadable embedding files ({type(e).__name__}: {e})"
            log.warning("Failed loading catalogue embeddings from %s: %s", self.brand_dir, e)
            self.items, self.embeddings = [], np.empty((0, 512), np.float32)
            return False
        for it in self.items:
            it.brand_slug = it.brand_slug or self.brand_dir.name
        if len(self.items) != len(self.embeddings):
            self.load_error = (f"manifest has {len(self.items)} records but embeddings.npz has {len(self.embeddings)} "
                               "rows; they no longer belong together")
            log.warning("%s: %s", self.brand_dir, self.load_error)
            return False
        log.info("Loaded %d embeddings for brand '%s'", len(self.items), self.brand)
        return True

    # ---- compatibility with the current model
    def compatibility(self, model_name: str, pretrained: str, preprocess: str | None = None) -> tuple[bool, str]:
        """Can these embeddings be compared with crops embedded by (model, pretrained, preprocess)?"""
        if self.load_error:
            return False, self.load_error
        if not len(self.items):
            return False, "the index is empty"
        m = self.meta
        if not m:      # index written before versions were recorded: it always used the default model
            if (model_name, pretrained) == (DEFAULT_MODEL, DEFAULT_PRETRAINED) and self.embeddings.shape[1] == 512:
                return True, "legacy index without version info (assumed ViT-B-32 / laion2b); rebuild once to record it"
            return False, (f"index has no version info and the configured model is {model_name}/{pretrained}; "
                           f"rebuild it")
        if (m.get("model"), m.get("pretrained")) != (model_name, pretrained):
            return False, (f"embeddings were made with {m.get('model')}/{m.get('pretrained')} but the configured model "
                           f"is {model_name}/{pretrained}")
        if m.get("dim") and m["dim"] != self.embeddings.shape[1]:
            return False, "embedding dimension does not match the recorded model"
        if preprocess and m.get("preprocess") and m["preprocess"] != preprocess:
            return False, "image preprocessing changed since the embeddings were made"
        return True, ""

    # ---- what is on disk vs what is indexed
    def scan(self) -> tuple[list[CatalogueItem], list[dict]]:
        """Every valid image of every product on disk, as items (not embedded). Also returns the products."""
        products = self.products()
        images_dir = self.brand_dir / "images"
        items: list[CatalogueItem] = []
        for p in products:
            handle = p.get("handle") or str(p.get("product_id"))
            folder = images_dir / self.brand_dir.name / handle
            if not folder.exists():
                folder = images_dir / handle
            if not folder.exists():
                continue
            cdn = p.get("image_urls") or []
            files = sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTS), key=lambda f: f.name)
            for i, f in enumerate(files):
                st = f.stat()
                items.append(CatalogueItem(
                    brand=p.get("brand") or self.brand, product_id=str(p.get("product_id") or handle), handle=handle,
                    name=p.get("name") or handle, category=p.get("category") or p.get("product_type") or "",
                    product_type=p.get("product_type") or "",
                    price=float(p["price"]) if p.get("price") is not None else None,
                    compare_at_price=float(p["compare_at_price"]) if p.get("compare_at_price") is not None else None,
                    currency=p.get("currency") or "EGP", url=p.get("url") or "",
                    image_path=f.relative_to(self.brand_dir).as_posix(),
                    image_rel=f.relative_to(images_dir).as_posix() if folder.parent != images_dir
                    else f"{self.brand_dir.name}/{handle}/{f.name}",
                    image_filename=f.name, image_url=_cdn_url_for(f.name, cdn, i), brand_slug=self.brand_dir.name,
                    image_size=st.st_size, image_mtime_ns=st.st_mtime_ns))
        return items, products

    def staleness(self) -> dict:
        """Images added / changed / removed on disk since the embeddings were made (cheap: file stats only)."""
        on_disk, _ = self.scan()
        indexed = {it.image_rel: it for it in self.items}
        disk = {it.image_rel: it for it in on_disk}
        new = [r for r in disk if r not in indexed]
        removed = [r for r in indexed if r not in disk]
        changed = [r for r, it in disk.items() if r in indexed and indexed[r].image_size is not None
                   and (indexed[r].image_size, indexed[r].image_mtime_ns) != (it.image_size, it.image_mtime_ns)]
        return {"new": len(new), "changed": len(changed), "removed": len(removed), "on_disk": len(disk)}

    # ---- building
    def build_or_update(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
                        device: str = "auto", batch_size: int = 64,
                        progress: Callable[[int, int], None] | None = None,
                        encoder=None, force: bool = False) -> int:
        """Embed images that are new or changed; keep every unchanged embedding. Returns how many were embedded.

        Everything is re-embedded when the model, its weights or the preprocessing differ from
        the ones recorded in embeddings_meta.json (or with force=True).
        """
        if not self.has_catalogue() or not (self.brand_dir / "images").exists():
            raise FileNotFoundError(f"Missing catalogue.jsonl or images/ in {self.brand_dir}")
        encoder = encoder or ClipEncoder(model_name, pretrained, device)
        meta = encoder.meta()
        old: dict[str, tuple[int, CatalogueItem]] = {}
        if not force and self.load():
            ok, why = self.compatibility(meta["model"], meta["pretrained"], meta["preprocess"])
            if ok:
                old = {it.image_rel: (i, it) for i, it in enumerate(self.items)}
                if not self.meta:
                    log.info("%s: adopting legacy index (no version info) without re-embedding", self.brand)
            else:
                log.info("%s: re-embedding everything: %s", self.brand, why)
        disk, _ = self.scan()
        keep_rows, keep_items, todo = [], [], []
        for it in disk:
            prev = old.get(it.image_rel)
            same_file = prev is not None and (prev[1].image_size is None
                                              or (prev[1].image_size, prev[1].image_mtime_ns) == (it.image_size, it.image_mtime_ns))
            if same_file:
                keep_rows.append(prev[0])
                keep_items.append(it)           # fresh product fields (name, price…) with the old vector
            else:
                todo.append(it)
        log.info("Brand '%s': %d images unchanged, %d to embed", self.brand, len(keep_items), len(todo))
        new_items, new_vecs = [], []
        for i in range(0, len(todo), batch_size):
            chunk, ims = [], []
            for it in todo[i:i + batch_size]:
                try:
                    with Image.open(self.brand_dir / it.image_path) as im:
                        ims.append(im.convert("RGB"))
                    chunk.append(it)
                except Exception as e:
                    log.warning("Skipping corrupt or unreadable catalogue image %s: %s", it.image_path, e)
            if ims:
                new_vecs.append(encoder.embed(ims, batch_size=batch_size))
                new_items += chunk
            if progress:
                progress(min(len(todo), i + batch_size), len(todo))
        parts = ([self.embeddings[keep_rows]] if keep_rows else []) + new_vecs
        if not parts:
            raise ValueError(f"No readable images found for brand '{self.brand}' in {self.brand_dir / 'images'}")
        self.embeddings = np.vstack(parts).astype(np.float32)
        self.items = keep_items + new_items
        self.meta = {"format": INDEX_FORMAT, **meta, "dim": int(self.embeddings.shape[1]), "images": len(self.items),
                     "products": len({it.product_id for it in self.items}),
                     "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        self._write()
        self.load_error, self._infos = None, None
        log.info("Saved %d embeddings for brand '%s' (%d new)", len(self.items), self.brand, len(new_items))
        return len(new_items)

    def _write(self) -> None:
        self.brand_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.embeddings_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, embeddings=self.embeddings)
        tmp.replace(self.embeddings_path)
        for path, data in ((self.manifest_path, [it.to_dict() for it in self.items]), (self.meta_path, self.meta)):
            t = path.with_suffix(".tmp.json")
            t.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            t.replace(path)

    # ---- taxonomy view of each indexed image
    def infos(self) -> list[ProductInfo]:
        if self._infos is None:
            tax = load_taxonomy()
            colour_opts: dict[str, list[str]] = {}
            try:
                for p in self.products():
                    colour_opts[str(p.get("product_id"))] = [v for o in p.get("options") or []
                                                             if str(o.get("name", "")).lower() in ("color", "colour")
                                                             for v in o.get("values") or []]
            except Exception:
                pass
            cache: dict[str, ProductInfo] = {}
            out = []
            for it in self.items:
                if it.product_id not in cache:
                    cache[it.product_id] = tax.describe_product(it.name, it.product_type, colour_opts.get(it.product_id))
                out.append(cache[it.product_id])
            self._infos = out
        return self._infos


def _cdn_url_for(filename: str, urls: list[str], position: int) -> str:
    stem = Path(filename).stem
    for u in urls:
        if stem and stem in u:
            return u
    return urls[position] if position < len(urls) else ""


# ---------------------------------------------------------------- all brands

class MultiBrandCatalogue:
    """Every brand found in the configured catalogue folders."""

    def __init__(self, roots: list[Path] | None = None):
        if roots is None:
            from sceneseen.config import catalogue_dirs, load_config

            roots = catalogue_dirs(load_config().catalogue)
        self.roots = [Path(r) for r in roots]
        self.indices: dict[str, CatalogueIndex] = {}

    def discover_and_load(self) -> None:
        for root in self.roots:
            if not root.exists():
                continue
            for brand_dir in sorted(root.iterdir()):
                if brand_dir.is_dir() and (brand_dir / "catalogue.jsonl").exists() and brand_dir.name not in self.indices:
                    idx = CatalogueIndex(brand=brand_dir.name, brand_dir=brand_dir)
                    idx.load()
                    self.indices[brand_dir.name] = idx

    def total_products(self) -> int:
        return sum(len({it.product_id for it in idx.items}) for idx in self.indices.values())

    def total_images(self) -> int:
        return sum(len(idx.items) for idx in self.indices.values())

    def status(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
               preprocess: str | None = None, check_files: bool = True) -> dict:
        """ready | missing_dataset | not_built | incompatible, with a message a person can act on."""
        if not self.indices:
            self.discover_and_load()
        roots = [str(r) for r in self.roots]
        if not self.indices:
            return {"status": "missing_dataset", "brands": [], "usable": [], "warnings": [], "searched": roots,
                    "products": 0, "images": 0,
                    "message": ("No product catalogue found. Put each brand folder (catalogue.jsonl + images/) in one of: "
                                + ", ".join(roots) + " — or set SCENESEEN_CATALOGUE_DIR to the folder that contains it.")}
        brands, usable, problems, warnings = [], [], [], []
        for slug, idx in self.indices.items():
            b = {"brand": slug, "dir": str(idx.brand_dir), "images": len(idx.items),
                 "products": len({it.product_id for it in idx.items})}
            if not idx.is_built():
                b["status"] = "not_built"
                problems.append(f"{slug}: embeddings have not been generated yet (run: {BUILD_HINT} --brand {slug})")
            else:
                ok, why = idx.compatibility(model_name, pretrained, preprocess)
                b["status"] = "ready" if ok else "incompatible"
                if ok:
                    usable.append(slug)
                    if why:
                        warnings.append(f"{slug}: {why}")
                    if check_files:
                        st = idx.staleness()
                        b["stale"] = st
                        if st["new"] or st["changed"] or st["removed"]:
                            warnings.append(f"{slug}: {st['new']} new, {st['changed']} changed, {st['removed']} removed "
                                            f"image(s) since the embeddings were made; run: {BUILD_HINT} --brand {slug}")
                else:
                    problems.append(f"{slug}: {why}. Regenerate with: {BUILD_HINT} --brand {slug}")
            brands.append(b)
        if usable:
            status, msg = "ready", ""
        elif all(b["status"] == "not_built" for b in brands):
            status, msg = "not_built", "Catalogue found but its embeddings are missing. " + " ".join(problems)
        else:
            status, msg = "incompatible", "Catalogue embeddings cannot be used. " + " ".join(problems)
        return {"status": status, "message": msg, "brands": brands, "usable": usable,
                "warnings": warnings + (problems if usable else []), "searched": roots,
                "products": sum(b["products"] for b in brands if b["brand"] in usable),
                "images": sum(b["images"] for b in brands if b["brand"] in usable)}

    def fingerprint(self) -> str:
        parts = [f"{s}:{len(i.items)}:{i.meta.get('built_at', '')}:{i.meta.get('model', '')}" for s, i in sorted(self.indices.items())]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]

    # ---- ranking (pure numpy; no model)
    def rank(self, query_vec: np.ndarray, query: DetectionQuery | None, top_k: int = 3,
             brands: list[str] | None = None, color_weights: dict | None = None) -> dict:
        """Top-k products for one embedded crop.

        query=None → unfiltered visual search (tools / tests). Otherwise only products whose
        normalised type is in query.strict_types are used; products of query.broad_types fill the
        remaining slots only if the strict pool has fewer than top_k products. Unknown-type
        products are never offered.
        """
        tax = load_taxonomy()
        w = {"same": 0.04, "near": 0.02, "different": -0.04, "different_multicolor": -0.01, **(color_weights or {})}
        active = [i for s, i in self.indices.items() if (brands is None or s in brands) and len(i.items)]
        if query and not query.strict_types and not query.broad_types:
            return {"candidates": [], "fallback": False, "pool": {"strict": 0, "broad": 0},
                    "note": "No product type in the catalogue corresponds to this kind of object."}
        tiers = [("strict", set(query.strict_types)), ("broad", set(query.broad_types))] if query else [("any", None)]
        pools: dict[str, dict] = {}
        for tier, allowed in tiers:
            best: dict[str, tuple[float, float, CatalogueItem, ProductInfo, str]] = {}
            for idx in active:
                infos = idx.infos()
                rows = np.arange(len(idx.items)) if allowed is None else \
                    np.array([i for i, inf in enumerate(infos) if inf.product_type in allowed], dtype=int)
                if not len(rows):
                    continue
                sims = idx.embeddings[rows] @ query_vec
                for r, s in zip(rows.tolist(), sims.tolist()):
                    it, inf = idx.items[r], infos[r]
                    key = f"{idx.brand_dir.name}:{it.product_id}"
                    if key not in best or s > best[key][0]:
                        rel = tax.color_relation(query.color if query else None, inf)
                        adj = {"same": w["same"], "near": w["near"], "unknown": 0.0,
                               "different": w["different_multicolor"] if inf.multicolor else w["different"]}[rel]
                        best[key] = (s, s + adj, it, inf, rel)
            pools[tier] = best
        out: list[ProductMatch] = []
        used: set[str] = set()
        fallback = False
        for tier, _ in tiers:
            if len(out) >= top_k:
                break
            if tier == "broad":
                if not pools.get("broad"):
                    break
                fallback = True
            ranked = sorted(((k, v) for k, v in pools.get(tier, {}).items() if k not in used), key=lambda kv: -kv[1][1])
            for k, (s, score, it, inf, rel) in ranked[:top_k - len(out)]:
                used.add(k)
                out.append(ProductMatch(
                    product_id=it.product_id, brand=it.brand, name=it.name, category=it.category,
                    product_type=it.product_type, price=it.price, compare_at_price=it.compare_at_price,
                    currency=it.currency, url=it.url, similarity=s, best_image_rel=it.image_rel,
                    best_image_url=it.image_url, brand_slug=it.brand_slug, rank_score=score,
                    normalized_type=inf.product_type, umbrella=inf.umbrella, colors=list(inf.colors),
                    color_relation=rel, match_tier="strict" if tier == "any" else tier))
        n_strict, n_broad = len(pools.get("strict", pools.get("any", {}))), len(pools.get("broad", {}))
        note = ""
        if fallback:
            note = (f"Only {n_strict} product(s) of the exact type; added the closest products from broader "
                    f"compatible categories.")
            log.info("matching fallback for '%s': strict pool %d < %d, broad pool %d",
                     query.label if query else "?", n_strict, top_k, n_broad)
        elif query and not out:
            note = "The catalogue has no products of a compatible type."
        return {"candidates": out, "fallback": fallback, "pool": {"strict": n_strict, "broad": n_broad}, "note": note}

    def match_crop(self, crop_image: Image.Image, model=None, preprocess=None, device: str = "cpu", top_k: int = 3,
                   brand_filter: str | None = None, label: str | None = None, detection_type: str | None = None,
                   color: str | None = None, encoder=None) -> list[ProductMatch]:
        """Match one crop. Without label/detection_type it is a raw (unfiltered) visual search."""
        if not self.indices:
            self.discover_and_load()
        if encoder is None:
            encoder = _ModelEncoder(model, preprocess, device)
        vec = encoder.embed([crop_image])[0]
        q = load_taxonomy().detection_query(label or "", detection_type, color) if (label or detection_type) else None
        return self.rank(vec, q, top_k, [brand_filter] if brand_filter else None)["candidates"]


class _ModelEncoder:
    """Wraps an already-loaded (model, preprocess) pair."""

    def __init__(self, model, preprocess, device):
        self.model, self.pre, self.device = model, preprocess, device

    def embed(self, images, batch_size: int = 64) -> np.ndarray:
        import torch

        t = torch.stack([self.pre(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad():
            e = self.model.encode_image(t).float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------- a whole video

def _crop(frame_file: Path, box: list[float], pad: float) -> Image.Image:
    with Image.open(frame_file) as im:
        im = im.convert("RGB")
        w, h = im.size
        x1, y1, x2, y2 = box
        px, py = pad * (x2 - x1) * w, pad * (y2 - y1) * h
        return im.crop((int(max(0, x1 * w - px)), int(max(0, y1 * h - py)),
                        int(min(w, x2 * w + px)), int(min(h, y2 * h + py))))


def _settings(cfg) -> tuple[str, str, str, int, float, dict]:
    if cfg is None:
        return DEFAULT_MODEL, DEFAULT_PRETRAINED, "auto", 3, 0.08, {}
    return (cfg.model, cfg.pretrained, cfg.device, cfg.top_k, cfg.crop_pad,
            {"same": cfg.color_same, "near": cfg.color_near, "different": cfg.color_different,
             "different_multicolor": cfg.color_different_multicolor})


def match_video_candidates(candidates: list[dict], frames_dir: Path, multi_brand: MultiBrandCatalogue | None = None,
                           model_name: str | None = None, pretrained: str | None = None, device: str | None = None,
                           top_k: int | None = None, encoder=None, cfg=None) -> dict:
    """Match every candidate (detected object) of a video. Returns
    {"status", "message", "warnings", "matches": {candidate key: {"query", "candidates", "fallback", "pool", "note"}}}.
    Never raises for a missing or incompatible dataset: the status says what to do instead.
    """
    from sceneseen.commercial.frames import frame_name

    m, p, d, k, pad, colour_w = _settings(cfg)
    model_name, pretrained, device, top_k = model_name or m, pretrained or p, device or d, top_k or k
    multi_brand = multi_brand or MultiBrandCatalogue()
    st = multi_brand.status(model_name, pretrained, check_files=False)
    if st["status"] != "ready":
        return {**st, "matches": {}}
    encoder = encoder or ClipEncoder(model_name, pretrained, device)
    st = multi_brand.status(model_name, pretrained, getattr(encoder, "preprocess_fingerprint", None), check_files=False)
    if st["status"] != "ready":
        return {**st, "matches": {}}
    tax = load_taxonomy()
    crops, cands = [], []
    for c in candidates:
        best = c.get("best") or {}
        if not c.get("key") or not best:
            continue
        f = frames_dir / frame_name(best.get("shot_id", 0), best.get("position", 0.5))
        if not f.exists():
            continue
        try:
            crops.append(_crop(f, best.get("box", [0, 0, 1, 1]), pad))
            cands.append(c)
        except Exception as e:
            log.warning("Failed loading/cropping frame for candidate %s: %s", c.get("key"), e)
    matches: dict[str, dict] = {}
    if crops:
        vecs = encoder.embed(crops)
        for c, v in zip(cands, vecs):
            q = tax.detection_query(c.get("label", ""), c.get("type_id"), c.get("color"))
            r = multi_brand.rank(v, q, top_k, st["usable"], colour_w)
            matches[c["key"]] = {
                "query": {"label": q.label, "detection_type": q.detection_type, "color": q.color, "umbrella": q.umbrella,
                          "strict_types": list(q.strict_types), "broad_types": list(q.broad_types), "notes": list(q.notes)},
                "candidates": [x.to_dict() for x in r["candidates"]], "fallback": r["fallback"], "pool": r["pool"],
                "note": r["note"]}
    return {**st, "matches": matches}


def _cache_fingerprint(multi_brand: MultiBrandCatalogue, cfg) -> str:
    tax = load_taxonomy()
    blob = json.dumps({"index": multi_brand.fingerprint(), "taxonomy": tax.version,
                       "taxonomy_file": hashlib.sha1(json.dumps(tax.data, sort_keys=True).encode()).hexdigest()[:10],
                       "cfg": asdict(cfg) if cfg is not None else None}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def get_or_compute_matches(cache_dir: Path, candidates: list[dict], frames_dir: Path,
                           multi_brand: MultiBrandCatalogue | None = None, allow_compute: bool = True,
                           encoder=None, cfg=None) -> dict:
    """Matches for a video, cached in data/cache/<vid>/commercial/matches.json. The cache is reused
    only while the catalogue index, the taxonomy and the matching settings are unchanged."""
    multi_brand = multi_brand or MultiBrandCatalogue()
    if not multi_brand.indices:
        multi_brand.discover_and_load()
    m, p = _settings(cfg)[:2]
    mf = Path(cache_dir) / "commercial" / "matches.json"
    fp = _cache_fingerprint(multi_brand, cfg)
    cached: dict = {}
    if mf.exists():
        try:
            d = json.loads(mf.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("format") == INDEX_FORMAT and d.get("fingerprint") == fp:
                cached = d.get("matches", {})
        except Exception as e:
            log.warning("Unreadable matches cache at %s: %s", mf, e)
    wanted = {c["key"] for c in candidates if c.get("key")}
    if wanted and wanted <= set(cached):
        st = multi_brand.status(m, p, check_files=False)
        return {**st, "cached": True, "matches": {k: cached[k] for k in wanted}}
    if not allow_compute:
        st = multi_brand.status(m, p, check_files=False)
        return {**st, "status": "not_computed" if st["status"] == "ready" else st["status"], "cached": True,
                "matches": {k: v for k, v in cached.items() if k in wanted}}
    todo = [c for c in candidates if c.get("key") not in cached]
    res = match_video_candidates(todo, frames_dir, multi_brand, encoder=encoder, cfg=cfg)
    if res["status"] == "ready":
        cached.update(res["matches"])
        try:
            mf.parent.mkdir(parents=True, exist_ok=True)
            tmp = mf.with_suffix(".tmp")
            tmp.write_text(json.dumps({"format": INDEX_FORMAT, "fingerprint": fp, "matches": cached}, indent=1,
                                      ensure_ascii=False), encoding="utf-8")
            tmp.replace(mf)
        except Exception as e:
            log.warning("Failed saving matches cache: %s", e)
    return {**res, "cached": False, "matches": {k: v for k, v in cached.items() if k in wanted}}


def main():
    """`python -m sceneseen.commercial.matching --build-index` (same as `python -m sceneseen match-catalogue`)."""
    import sys

    from sceneseen.__main__ import main as cli

    return cli(["match-catalogue", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
