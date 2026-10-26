# SceneSeen — Brand Catalogue Scraper

This folder contains a reusable, brand-agnostic system for scraping clothing brand catalogues into structured product records (JSONL + CSV) with downloaded images.

Built as a data-collection layer for SceneSeen Phase 2B+: product catalogue matching and brand/product identification from video content.

---

## Directory Layout

```


---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BrandScraper                         │
│                    (scraper/base.py)                        │
│                                                             │
│  fetch_product_urls()  →  parse_product()  →  run()        │
│                                   ↓                         │
│              Product dataclass (canonical record)           │
│              download_images()                              │
│              append_jsonl() / write_csv()                   │
└──────────────────┬──────────────────────────────────────────┘
                   │ subclasses
       ┌───────────┴────────────┐
       │                        │
┌──────▼────────┐    ┌──────────▼──────────┐
│ ShopifyScraper│    │  (future scrapers)   │
│ shopify.py    │    │  e.g. WooCommerce,   │
│               │    │  custom HTML, etc.   │
│ /products.json│    └─────────────────────┘
│ pagination    │
└──────┬────────┘
       │ brand-specific config
┌──────▼──────────────┐
│  brands/            │
│  inyourshoe.py      │  ← sets brand_name, base_url, currency, delay
│  (future brands...) │
└─────────────────────┘
```

### Key design decisions

- **`BrandScraper` is abstract** — adding a brand means writing one small file in `brands/`, not touching shared infrastructure.
- **`ShopifyScraper` is brand-agnostic** — any public Shopify store can be scraped by passing `base_url`. No auth required; Shopify exposes `/products.json` publicly.
- **JSONL is the source of truth** — append-only, one record per line. The CSV is always regenerated from the JSONL at the end of each run.
- **`product_id` is the dedup key** — re-running or resuming after a crash is always safe; already-scraped IDs are read from the JSONL on startup and skipped.
- **Images are idempotent** — if a file already exists on disk it is not re-downloaded.
- **Retry logic is shared** — the HTTP session in `base.py` handles 429/5xx with exponential back-off automatically for all scrapers.

---

## Product Record Schema

Every scraped product is stored with these fields:

| Field | Type | Description |
|---|---|---|
| `brand` | str | Brand name (e.g. `"IN YOUR SHOE"`) |
| `product_id` | str | Platform product ID — used as dedup key |
| `handle` | str | URL slug (e.g. `black-oversized-leather-jacket`) |
| `name` | str | Product title |
| `description` | str | Plain-text description (HTML stripped) |
| `category` | str | Inferred from `product_type` or tags |
| `product_type` | str | Raw platform product type |
| `tags` | list[str] | Platform tags |
| `vendor` | str | Vendor/brand string from platform |
| `url` | str | Canonical product page URL |
| `price` | float | Min variant price |
| `compare_at_price` | float\|null | Pre-sale price, if any |
| `currency` | str | Currency code (e.g. `"EGP"`) |
| `available` | bool | `true` if any variant is in stock |
| `variants` | list[dict] | `{id, title, sku, price, compare_at_price, available, option1-3, grams, position}` |
| `options` | list[dict] | `{name, values}` — e.g. `{name: "Size", values: ["S","M","L","XL"]}` |
| `image_urls` | list[str] | All product image CDN URLs (deduplicated) |
| `local_images` | list[str] | Absolute paths to downloaded image files |
| `published_at` | str | ISO 8601 |
| `created_at` | str | ISO 8601 |
| `updated_at` | str | ISO 8601 |

---

## Scraped Brands

| Brand | Platform | Base URL | Currency | Products | Status |
|---|---|---|---|---|---|
| IN YOUR SHOE | Shopify | https://inyourshoe.com | EGP | 1,183 | ✅ Done |

---

## How to Run

> Activate the project venv first from the repo root:
> ```bash
> source .venv/bin/activate
> ```

### IN YOUR SHOE

```bash
# Quick test — 10 products
python -m scraper.brands.inyourshoe --test

# Full catalogue (safe to re-run; resumes from last position)
python -m scraper.brands.inyourshoe

# First N products only
python -m scraper.brands.inyourshoe --limit 50

# Text data only, skip image downloads
python -m scraper.brands.inyourshoe --no-images

# Custom output directory
python -m scraper.brands.inyourshoe --output scraper/output

# Verbose logging
python -m scraper.brands.inyourshoe --log-level DEBUG
```

---

## How to Add a New Brand

### Case A — Shopify store (< 10 lines)

Create `scraper/brands/newbrand.py`:

```python
from scraper.shopify import ShopifyScraper
from pathlib import Path
import argparse, logging, json

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    scraper = ShopifyScraper(
        brand_name="New Brand",
        base_url="https://newbrand.com",   # must be a Shopify store
        currency="USD",
        page_delay=1.2,
    )
    summary = scraper.run(Path("scraper/output"), limit=None)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
```

Run it: `python -m scraper.brands.newbrand`

That's it. Pagination, dedup, image download, JSONL, CSV — all inherited.

---

### Case B — Non-Shopify / custom site

Create `scraper/brands/mybrand.py` and subclass `BrandScraper`:

```python
from scraper.base import BrandScraper, Product
import requests

class MyBrandScraper(BrandScraper):

    @property
    def brand_name(self) -> str:
        return "My Brand"

    def fetch_product_urls(self, session: requests.Session) -> list[str]:
        """Crawl catalogue/collection pages and return all product URLs."""
        urls = []
        page = 1
        while True:
            resp = session.get(f"https://mybrand.com/shop?page={page}")
            # ... parse HTML for product links ...
            if not new_urls:
                break
            urls.extend(new_urls)
            page += 1
        return urls

    def parse_product(self, url: str, session: requests.Session) -> Product | None:
        """Fetch a single product page and return a Product record."""
        resp = session.get(url)
        # ... parse HTML ...
        return Product(
            brand=self.brand_name,
            product_id=...,
            name=...,
            description=...,
            price=...,
            image_urls=[...],
            url=url,
        )

if __name__ == "__main__":
    from pathlib import Path
    import logging
    logging.basicConfig(level=logging.INFO)
    MyBrandScraper().run(Path("scraper/output"))
```

Everything else — HTTP retries, dedup, image downloading, JSONL appending, CSV writing, summary reporting — is handled by the base class.

---

## Reliability Features

| Feature | How it works |
|---|---|
| **Pagination** | Loops until an empty page is returned (Shopify); easily adapted for cursor/offset pagination |
| **Deduplication** | `product_id` set loaded from JSONL at startup; duplicates are skipped silently |
| **Resumability** | JSONL is append-only; re-running after a crash picks up from where it left off |
| **HTTP retries** | 4 retries with 1.5× exponential back-off on 429/500/502/503/504 |
| **Image idempotency** | File-existence check before every download; no redundant requests |
| **Failure isolation** | Per-product try/except; one bad product never aborts the run |
| **Rate limiting** | Configurable `delay` between products; `page_delay` between catalogue pages |

---

## Relation to SceneSeen Pipeline

```
[This scraper]  →  scraper/output/<brand>/catalogue.jsonl
                                         /images/
                        ↓
            Phase 2B+ (not yet built):
            Product catalogue matching
            — given a detected object in a video frame,
              find the closest matching product record
              using image embeddings + metadata filters
```

The JSONL schema and image folder layout are designed to be directly consumed by a future embedding/matching pipeline — `product_id` as the lookup key, `local_images` as the paths to embed with CLIP.
