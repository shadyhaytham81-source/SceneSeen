"""CatalogueImporter (FR15): one subclass per file format, as in the class diagram.

parse() is real for all three formats. index() embeds each product's image and adds it to
the store. STUB part: images are embedded from their URL string, not downloaded; milestone 3
downloads each image, crops the garment (sceneseen.commercial.garment_crop) and embeds it.
"""
from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod

from ..schemas import ImportReport, Product
from ..store import Store
from .embedder import Embedder


def _price(text: str | None) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


class CatalogueImporter(ABC):
    format = ""

    def __init__(self, store: Store, embedder: Embedder):
        self.store, self.embedder = store, embedder

    @abstractmethod
    def row_to_product(self, row: dict, brand_id: str) -> Product | None:
        """One CSV row -> Product, or None to skip it (e.g. Shopify variant rows)."""

    def parse(self, data: bytes, brand_id: str) -> tuple[list[Product], list[int]]:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        products, bad = [], []
        for n, row in enumerate(reader, start=1):
            try:
                p = self.row_to_product({k.strip(): (v or "").strip() for k, v in row.items() if k}, brand_id)
            except (KeyError, ValueError):
                bad.append(n)
                continue
            if p:
                products.append(p)
        return products, bad

    def index(self, products: list[Product]) -> int:
        for p in products:
            self.store.products[p.id] = p
            self.store.product_vectors[p.id] = self.embedder.embed((p.image_url or p.product_url).encode())
        return len(products)

    def run(self, data: bytes, brand_id: str) -> ImportReport:
        products, bad = self.parse(data, brand_id)
        return ImportReport(brand_id=brand_id, format=self.format, imported=self.index(products), skipped_rows=bad)


class ShopifyImporter(CatalogueImporter):
    """Shopify product export. Rows without a Title are extra variants/images: skipped."""
    format = "shopify"

    def __init__(self, store: Store, embedder: Embedder, store_url: str = ""):
        super().__init__(store, embedder)
        self.store_url = store_url.rstrip("/")

    def row_to_product(self, row, brand_id):
        if not row.get("Title"):
            return None
        if not row["Handle"]:
            raise ValueError("no handle")
        return Product(brand_id=brand_id, title=row["Title"], category=row.get("Type", "").lower() or "unknown",
                       price=_price(row.get("Variant Price")), image_url=row.get("Image Src") or None,
                       product_url=f"{self.store_url}/products/{row['Handle']}")


class WooCommerceImporter(CatalogueImporter):
    """WooCommerce product export (Products > Export)."""
    format = "woocommerce"

    def row_to_product(self, row, brand_id):
        if row.get("Type", "simple") == "variation":
            return None
        url = row.get("Permalink") or row.get("External URL")
        if not row["Name"] or not url:
            raise ValueError("missing name or url")
        category = row.get("Categories", "").split(">")[-1].split(",")[0].strip().lower()
        return Product(brand_id=brand_id, title=row["Name"], category=category or "unknown",
                       price=_price(row.get("Sale price") or row.get("Regular price")),
                       image_url=(row.get("Images", "").split(",")[0].strip() or None), product_url=url)


class TemplateImporter(CatalogueImporter):
    """Our template for brands without Shopify/WooCommerce: title, category, price, product_url, image_url, colour."""
    format = "template"

    def row_to_product(self, row, brand_id):
        if not row["title"] or not row["product_url"] or not row["category"]:
            raise ValueError("missing required column")
        return Product(brand_id=brand_id, title=row["title"], category=row["category"].lower(),
                       price=_price(row.get("price")), product_url=row["product_url"],
                       image_url=row.get("image_url") or None, colour=row.get("colour") or None)


IMPORTERS = {c.format: c for c in (ShopifyImporter, WooCommerceImporter, TemplateImporter)}
