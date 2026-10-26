"""
scraper/base.py
---------------
Abstract base class every brand scraper must implement.
Adding a new brand = subclass BrandScraper, implement the three abstract methods.
"""

from __future__ import annotations

import abc
import csv
import hashlib
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ─────────────────────────── data model ────────────────────────────

class Product:
    """Canonical product record used across all scrapers."""

    __slots__ = (
        "brand", "product_id", "handle", "name", "description",
        "category", "product_type", "tags", "vendor",
        "url", "price", "compare_at_price", "currency",
        "available", "variants", "options",
        "image_urls", "local_images",
        "published_at", "created_at", "updated_at",
        "raw",
    )

    def __init__(self, **kwargs: Any):
        for s in self.__slots__:
            setattr(self, s, kwargs.get(s))
        self.variants: list[dict] = kwargs.get("variants") or []
        self.options: list[dict] = kwargs.get("options") or []
        self.image_urls: list[str] = kwargs.get("image_urls") or []
        self.local_images: list[str] = kwargs.get("local_images") or []
        self.tags: list[str] = kwargs.get("tags") or []
        self.raw: dict = kwargs.get("raw") or {}

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__ if s != "raw"}

    def csv_row(self) -> dict:
        """Flat representation suitable for CSV export."""
        d = self.to_dict()
        d["variants"] = json.dumps(d["variants"])
        d["options"] = json.dumps(d["options"])
        d["image_urls"] = "|".join(d["image_urls"] or [])
        d["local_images"] = "|".join(d["local_images"] or [])
        d["tags"] = "|".join(d["tags"] or [])
        return d


CSV_FIELDS = [
    "brand", "product_id", "handle", "name", "description",
    "category", "product_type", "tags", "vendor",
    "url", "price", "compare_at_price", "currency",
    "available", "variants", "options",
    "image_urls", "local_images",
    "published_at", "created_at", "updated_at",
]


# ─────────────────────────── HTTP session ──────────────────────────

def make_session(
    retries: int = 4,
    backoff: float = 1.5,
    timeout: int = 20,
) -> requests.Session:
    """Shared session with retry / back-off logic."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "SceneSeen-Catalogue-Scraper/1.0 "
            "(research project; contact: sceneseen@project.local)"
        ),
        "Accept": "application/json, text/html, */*",
    })
    session._default_timeout = timeout
    return session


# ─────────────────────────── image downloader ──────────────────────

def _url_to_filename(url: str, position: int = 0) -> str:
    """Derive a clean local filename from an image URL."""
    parsed = urlparse(url)
    name = Path(parsed.path).name           # e.g. my-product.jpg?v=123
    name = re.sub(r"\?.*$", "", name)       # strip query string
    name = re.sub(r"[^\w.\-]", "_", name)  # sanitise
    if not name or name == ".":
        digest = hashlib.md5(url.encode()).hexdigest()[:12]
        name = f"img_{digest}_{position}.jpg"
    return name


def download_images(
    product: Product,
    images_dir: Path,
    session: requests.Session,
    delay: float = 0.3,
) -> list[str]:
    """
    Download all images for *product* into *images_dir/brand/handle/*.
    Returns list of relative paths that were actually saved.
    """
    brand_slug = re.sub(r"[^\w\-]", "_", (product.brand or "unknown").lower())
    dest = images_dir / brand_slug / (product.handle or product.product_id or "unknown")
    dest.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for i, url in enumerate(product.image_urls or []):
        filename = _url_to_filename(url, i)
        filepath = dest / filename
        rel_path = str(filepath)

        if filepath.exists():
            saved.append(rel_path)
            continue

        try:
            resp = session.get(url, timeout=session._default_timeout, stream=True)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            saved.append(rel_path)
            logger.debug("  ✓ image %s", filename)
        except Exception as exc:
            logger.warning("  ✗ image download failed for %s: %s", url, exc)

        time.sleep(delay)

    return saved


# ─────────────────────────── output writers ────────────────────────

def append_jsonl(product: Product, jsonl_path: Path) -> None:
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(product.to_dict(), ensure_ascii=False) + "\n")


def write_csv(products: list[Product], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for p in products:
            writer.writerow(p.csv_row())


# ─────────────────────────── base class ────────────────────────────

class BrandScraper(abc.ABC):
    """
    Subclass this for each brand.  Implement:
      - brand_name  (str property)
      - fetch_product_urls()  → iterator of product-page URLs
      - parse_product(url, session)  → Product
    """

    # ── override in subclass ──────────────────────────────────────
    @property
    @abc.abstractmethod
    def brand_name(self) -> str: ...

    @abc.abstractmethod
    def fetch_product_urls(self, session: requests.Session) -> list[str]: ...

    @abc.abstractmethod
    def parse_product(self, url: str, session: requests.Session) -> Product | None: ...

    # ── shared run logic ─────────────────────────────────────────
    def run(
        self,
        output_dir: Path,
        *,
        limit: int | None = None,
        download_images_flag: bool = True,
        delay: float = 0.5,
    ) -> dict:
        """
        Main entry point.  Returns a summary dict.
        Call with limit=10 for a test run, limit=None for full catalogue.
        """
        output_dir = Path(output_dir)
        brand_dir = output_dir / re.sub(r"[^\w\-]", "_", self.brand_name.lower())
        brand_dir.mkdir(parents=True, exist_ok=True)

        images_dir = brand_dir / "images"
        images_dir.mkdir(exist_ok=True)

        jsonl_path = brand_dir / "catalogue.jsonl"
        csv_path   = brand_dir / "catalogue.csv"

        # Load already-scraped IDs to avoid duplicates on resume
        seen_ids: set[str] = set()
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if rec.get("product_id"):
                        seen_ids.add(str(rec["product_id"]))
                except json.JSONDecodeError:
                    pass
            logger.info("Resuming — %d products already in JSONL.", len(seen_ids))

        session = make_session()

        # ── fetch URL list ───────────────────────────────────────
        logger.info("[%s] Fetching product list…", self.brand_name)
        urls = self.fetch_product_urls(session)
        total_available = len(urls)
        if limit:
            urls = urls[:limit]
        logger.info("  %d URLs fetched; processing %d.", total_available, len(urls))

        # ── scrape each product ──────────────────────────────────
        scraped: list[Product] = []
        failed: list[str] = []

        for i, url in enumerate(urls, 1):
            try:
                product = self.parse_product(url, session)
            except Exception as exc:
                logger.warning("  [%d/%d] FAIL %s — %s", i, len(urls), url, exc)
                failed.append(url)
                continue

            if product is None:
                logger.warning("  [%d/%d] SKIP (None) %s", i, len(urls), url)
                continue

            pid = str(product.product_id or "")
            if pid and pid in seen_ids:
                logger.debug("  [%d/%d] DUP  %s", i, len(urls), pid)
                continue

            if pid:
                seen_ids.add(pid)

            # Download images
            if download_images_flag and product.image_urls:
                product.local_images = download_images(
                    product, images_dir, session, delay=delay * 0.5
                )

            append_jsonl(product, jsonl_path)
            scraped.append(product)
            logger.info(
                "  [%d/%d] ✓ %s (%d images)",
                i, len(urls), product.name, len(product.image_urls or [])
            )
            time.sleep(delay)

        # ── write CSV from full JSONL (includes previous runs) ───
        all_products: list[Product] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                all_products.append(Product(**{k: rec.get(k) for k in Product.__slots__ if k != "raw"}))
            except Exception:
                pass
        write_csv(all_products, csv_path)

        summary = {
            "brand": self.brand_name,
            "total_in_catalogue": total_available,
            "processed_this_run": len(urls),
            "scraped_this_run": len(scraped),
            "failed_this_run": len(failed),
            "total_in_jsonl": len(all_products),
            "jsonl_path": str(jsonl_path),
            "csv_path": str(csv_path),
            "images_dir": str(images_dir),
            "failed_urls": failed,
        }
        logger.info("Done. %s", summary)
        return summary
