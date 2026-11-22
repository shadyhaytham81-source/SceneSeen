"""
scraper/brands/novencci.py
----------------------------
Scraper for https://novenccieg.com  (Shopify store).
Currency: EGP (Egyptian Pounds — prices shown on the site).

Usage:
    python -m scraper.brands.novencci              # full catalogue
    python -m scraper.brands.novencci --test       # 10-product test
    python -m scraper.brands.novencci --limit 50   # custom limit
    python -m scraper.brands.novencci --no-images  # skip image downloads
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from scraper.shopify import ShopifyScraper

BRAND_NAME = "NOVENCCI"
BASE_URL   = "https://novenccieg.com"
CURRENCY   = "EGP"


def build_scraper() -> ShopifyScraper:
    return ShopifyScraper(
        brand_name=BRAND_NAME,
        base_url=BASE_URL,
        currency=CURRENCY,
        page_delay=1.2,   # polite gap between catalogue pages
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape the NOVENCCI catalogue.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test",  action="store_true", help="Quick 10-product test run")
    group.add_argument("--limit", type=int, metavar="N",  help="Scrape first N products")
    parser.add_argument("--no-images", action="store_true", help="Skip image downloads")
    parser.add_argument(
        "--output", default="scraper/output",
        help="Root output directory (default: scraper/output)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    limit = 10 if args.test else args.limit  # None → full catalogue

    # Resolve output dir relative to the repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = repo_root / args.output

    scraper = build_scraper()
    summary = scraper.run(
        output_dir=output_dir,
        limit=limit,
        download_images_flag=not args.no_images,
        delay=0.6,
    )

    print("\n" + "═" * 60)
    print("  SCRAPE SUMMARY — NOVENCCI")
    print("═" * 60)
    print(f"  Total products in catalogue : {summary['total_in_catalogue']}")
    print(f"  Processed this run          : {summary['processed_this_run']}")
    print(f"  Scraped (new) this run      : {summary['scraped_this_run']}")
    print(f"  Failed this run             : {summary['failed_this_run']}")
    print(f"  Total records in JSONL      : {summary['total_in_jsonl']}")
    print(f"  JSONL output                : {summary['jsonl_path']}")
    print(f"  CSV  output                 : {summary['csv_path']}")
    print(f"  Images directory            : {summary['images_dir']}")
    if summary["failed_urls"]:
        print(f"\n  ⚠ Failed URLs ({len(summary['failed_urls'])}):")
        for u in summary["failed_urls"]:
            print(f"    - {u}")
    print("═" * 60)

    # Write run summary JSON alongside the catalogue
    brand_dir = Path(summary["jsonl_path"]).parent
    summary_path = brand_dir / "last_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  Run summary saved to        : {summary_path}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
