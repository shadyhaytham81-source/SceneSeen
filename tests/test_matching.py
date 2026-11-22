"""Tests for Phase 2B Product Catalogue Matching."""
import json
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from sceneseen.commercial.matching import (
    CatalogueIndex,
    CatalogueItem,
    MultiBrandCatalogue,
    ProductMatch,
)
from server.app import app


@pytest.fixture
def dummy_brand_dir(tmp_path):
    brand_dir = tmp_path / "test_brand"
    brand_dir.mkdir()
    images_dir = brand_dir / "images" / "test_brand"
    images_dir.mkdir(parents=True)

    # Create 3 products with dummy images
    products = []
    for i in range(3):
        handle = f"product-{i}"
        pdir = images_dir / handle
        pdir.mkdir(parents=True)

        img_paths = []
        for j in range(2):
            img_file = pdir / f"img_{j}.jpg"
            # Create a simple 32x32 colored image
            color = (50 * i, 70 * j, 100)
            im = Image.new("RGB", (32, 32), color=color)
            im.save(img_file)
            img_paths.append(str(img_file))

        products.append({
            "product_id": f"pid_{i}",
            "handle": handle,
            "name": f"Test Product {i}",
            "brand": "Test Brand",
            "category": "T-Shirts",
            "product_type": "T-Shirts",
            "price": 199.0 + i * 50,
            "currency": "EGP",
            "url": f"https://example.com/products/{handle}",
            "image_urls": [f"https://cdn.example.com/{handle}/img_{j}.jpg" for j in range(2)],
        })

    cat_file = brand_dir / "catalogue.jsonl"
    cat_file.write_text("\n".join(json.dumps(p) for p in products) + "\n")
    return brand_dir


def test_catalogue_index_build_and_load(dummy_brand_dir):
    idx = CatalogueIndex(brand="test_brand", brand_dir=dummy_brand_dir)
    assert not idx.is_built()

    # Build index
    count = idx.build_or_update(batch_size=2)
    assert count == 6  # 3 products * 2 images = 6 images
    assert idx.is_built()
    assert len(idx.items) == 6
    assert idx.embeddings.shape == (6, 512)

    # Verify normalization (L2 norm should be ~1.0)
    norms = np.linalg.norm(idx.embeddings, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # Load from persistent disk
    idx2 = CatalogueIndex(brand="test_brand", brand_dir=dummy_brand_dir)
    assert idx2.load()
    assert len(idx2.items) == 6
    assert idx2.items[0].product_id == "pid_0"
    assert idx2.items[0].brand == "Test Brand"
    assert idx2.items[0].name == "Test Product 0"
    assert idx2.items[0].price == 199.0
    assert idx2.items[0].currency == "EGP"

    # Incremental update should do 0 new embeddings
    assert idx2.build_or_update() == 0


def test_corrupted_image_graceful_handling(tmp_path):
    brand_dir = tmp_path / "bad_brand"
    brand_dir.mkdir()
    images_dir = brand_dir / "images" / "bad_brand" / "prod"
    images_dir.mkdir(parents=True)

    # Write a corrupted non-image file
    bad_img = images_dir / "corrupted.jpg"
    bad_img.write_text("not a real image")

    # Write a good image
    good_img = images_dir / "good.jpg"
    Image.new("RGB", (32, 32), color="blue").save(good_img)

    cat_file = brand_dir / "catalogue.jsonl"
    cat_file.write_text(json.dumps({
        "product_id": "bad_pid",
        "handle": "prod",
        "name": "Corrupt Test",
        "brand": "Bad Brand",
    }) + "\n")

    idx = CatalogueIndex(brand="bad_brand", brand_dir=brand_dir)
    count = idx.build_or_update()
    # Only the 1 good image should be embedded without crashing
    assert count == 1
    assert len(idx.items) == 1
    assert idx.items[0].image_filename == "good.jpg"


def test_multibrand_matching_and_deduplication(dummy_brand_dir):
    idx = CatalogueIndex(brand="test_brand", brand_dir=dummy_brand_dir)
    idx.build_or_update()

    mb = MultiBrandCatalogue(roots=[dummy_brand_dir.parent])
    mb.discover_and_load()
    assert mb.total_products() == 3
    assert mb.total_images() == 6

    # Match with a dummy crop
    query_crop = Image.new("RGB", (64, 64), color="blue")
    from sceneseen.features import load_clip
    model, preprocess = load_clip("ViT-B-32", "laion2b_s34b_b79k", "cpu")

    matches = mb.match_crop(query_crop, model, preprocess, device="cpu", top_k=2)
    assert len(matches) == 2
    # Ensure distinct products
    assert matches[0].product_id != matches[1].product_id
    # Scores descending
    assert matches[0].similarity >= matches[1].similarity


def test_api_catalogue_image_endpoint(dummy_brand_dir, monkeypatch):
    """Images are served only from the configured catalogue folders (no real dataset needed)."""
    monkeypatch.setenv("SCENESEEN_CATALOGUE_DIR", str(dummy_brand_dir.parent))
    client = TestClient(app)
    rel = "test_brand/product-1/img_0.jpg"
    r = client.get(f"/api/catalogue/image?rel={rel}&brand=test_brand")
    assert r.status_code == 200 and len(r.content) > 100
    r_thumb = client.get(f"/api/catalogue/image?rel={rel}&brand=test_brand&size=16")
    assert r_thumb.status_code == 200 and r_thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/catalogue/image?rel={rel}").status_code == 200          # brand taken from the path
    # path traversal and lookups outside the catalogue folders
    for bad in ("../../etc/passwd", "test_brand/../../catalogue.jsonl", "/etc/passwd"):
        assert client.get("/api/catalogue/image", params={"rel": bad, "brand": "test_brand"}).status_code in (400, 404)
    assert client.get("/api/catalogue/image", params={"rel": "x.jpg", "brand": "../x"}).status_code == 400
    assert client.get("/api/catalogue/image?rel=test_brand/does_not_exist.jpg").status_code == 404


def test_api_catalogue_status_reports_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENESEEN_CATALOGUE_DIR", str(tmp_path / "nothing-here"))
    r = TestClient(app).get("/api/catalogue/status")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "missing_dataset" and "SCENESEEN_CATALOGUE_DIR" in d["message"]


def test_api_matches_unknown_video_is_404():
    assert TestClient(app).get("/api/videos/0123456789abcdef/commercial/matches").status_code == 404
