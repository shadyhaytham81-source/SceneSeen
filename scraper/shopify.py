"""
scraper/shopify.py
------------------
Generic Shopify-powered brand scraper.
Reads /products.json (no auth needed; public Shopify API).

Usage for any Shopify store:
    scraper = ShopifyScraper(
        brand_name="My Brand",
        base_url="https://example.myshopify.com",
        currency="EGP",           # optional, falls back to price string
    )
    scraper.run(Path("data/catalogues"), limit=10)
"""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import requests

from .base import BrandScraper, Product, make_session

logger = logging.getLogger(__name__)


def _strip_html(raw_html: str | None) -> str:
    """Remove HTML tags and decode entities, return clean plain text."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ShopifyScraper(BrandScraper):
    """
    Works with any public Shopify store.
    Pagination: /products.json?limit=250&page=N (stops when empty page returned).
    """

    def __init__(
        self,
        brand_name: str,
        base_url: str,
        currency: str = "",
        page_delay: float = 1.0,
    ):
        self._brand_name = brand_name.strip()
        self._base_url = base_url.rstrip("/")
        self._currency = currency
        self._page_delay = page_delay

    @property
    def brand_name(self) -> str:
        return self._brand_name

    # ── catalogue fetch ──────────────────────────────────────────

    def _iter_raw_products(
        self, session: requests.Session
    ) -> Iterator[dict]:
        """Paginate /products.json and yield raw product dicts."""
        page = 1
        while True:
            url = f"{self._base_url}/products.json?limit=250&page={page}"
            try:
                resp = session.get(url, timeout=session._default_timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("Failed fetching page %d: %s", page, exc)
                break

            products = data.get("products", [])
            if not products:
                break

            logger.debug("  Page %d → %d products", page, len(products))
            yield from products
            page += 1
            time.sleep(self._page_delay)

    def fetch_product_urls(self, session: requests.Session) -> list[str]:
        """
        Return synthetic product-page URLs derived from handles.
        We actually parse directly from the JSON API so fetch_product_urls
        is used only for the base-class URL-list interface.
        The real data is in _raw_products (populated as a side effect).
        """
        self._raw_products: dict[str, dict] = {}
        urls: list[str] = []
        for raw in self._iter_raw_products(session):
            handle = raw.get("handle", "")
            url = f"{self._base_url}/products/{handle}"
            self._raw_products[url] = raw
            urls.append(url)
        return urls

    # ── product parsing ──────────────────────────────────────────

    def parse_product(self, url: str, session: requests.Session) -> Product | None:
        """Parse a product from the already-fetched raw JSON dict."""
        raw = self._raw_products.get(url)
        if raw is None:
            logger.warning("No cached raw for URL %s — skipping.", url)
            return None

        return self._raw_to_product(raw, url)

    def _raw_to_product(self, raw: dict, url: str) -> Product:
        # ── basic fields ───────────────────────────────────────
        product_id = str(raw.get("id", ""))
        handle     = raw.get("handle", "")
        name       = raw.get("title", "")
        description = _strip_html(raw.get("body_html", ""))
        product_type = raw.get("product_type", "")
        vendor       = raw.get("vendor", "")
        tags         = raw.get("tags") or []
        published_at = raw.get("published_at", "")
        created_at   = raw.get("created_at", "")
        updated_at   = raw.get("updated_at", "")

        # ── price / availability (from variants) ───────────────
        variants_raw = raw.get("variants") or []
        prices = [
            float(v["price"]) for v in variants_raw
            if v.get("price") not in (None, "")
        ]
        compare_prices = [
            float(v["compare_at_price"])
            for v in variants_raw
            if v.get("compare_at_price") not in (None, "")
        ]
        price            = min(prices) if prices else None
        compare_at_price = min(compare_prices) if compare_prices else None
        available        = any(v.get("available", False) for v in variants_raw)

        # ── structured variants ────────────────────────────────
        variants = [
            {
                "id":               str(v.get("id", "")),
                "title":            v.get("title", ""),
                "sku":              v.get("sku", ""),
                "price":            v.get("price"),
                "compare_at_price": v.get("compare_at_price"),
                "available":        v.get("available", False),
                "option1":          v.get("option1"),
                "option2":          v.get("option2"),
                "option3":          v.get("option3"),
                "grams":            v.get("grams"),
                "position":         v.get("position"),
            }
            for v in variants_raw
        ]

        # ── options (e.g. Size, Color) ─────────────────────────
        options = [
            {"name": o.get("name", ""), "values": o.get("values", [])}
            for o in (raw.get("options") or [])
        ]

        # ── images ────────────────────────────────────────────
        images_raw = raw.get("images") or []
        # Clean CDN URLs: strip query param for a consistent key but keep full URL
        image_urls = [img["src"] for img in images_raw if img.get("src")]
        # Deduplicate while preserving order
        seen_imgs: set[str] = set()
        unique_image_urls: list[str] = []
        for img_url in image_urls:
            base_url_img = re.sub(r"\?.*$", "", img_url)
            if base_url_img not in seen_imgs:
                seen_imgs.add(base_url_img)
                unique_image_urls.append(img_url)

        # ── category: use product_type or fall back to tags ───
        category = product_type or next(
            (t for t in tags if not t[0].isdigit()), ""
        )

        return Product(
            brand=self._brand_name,
            product_id=product_id,
            handle=handle,
            name=name,
            description=description,
            category=category,
            product_type=product_type,
            tags=tags,
            vendor=vendor,
            url=url,
            price=price,
            compare_at_price=compare_at_price,
            currency=self._currency,
            available=available,
            variants=variants,
            options=options,
            image_urls=unique_image_urls,
            local_images=[],
            published_at=published_at,
            created_at=created_at,
            updated_at=updated_at,
            raw=raw,
        )
