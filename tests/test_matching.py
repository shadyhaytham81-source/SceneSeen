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
    get_or_compute_matches,
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


def test_api_catalogue_image_endpoint():
    client = TestClient(app)

    # Existing IN YOUR SHOE image test
    rel = "in_your_shoe/black-oversized-leather-jacket/black-oversized-leather-jacket-leather-jacket-in-your-shoe-513016.jpg"
    r = client.get(f"/api/catalogue/image?rel={rel}&brand=in_your_shoe")
    assert r.status_code == 200
    assert r.headers["content-type"] in ("image/jpeg", "image/jpg")
    assert len(r.content) > 1000

    # Test thumbnail resize
    r_thumb = client.get(f"/api/catalogue/image?rel={rel}&brand=in_your_shoe&size=120")
    assert r_thumb.status_code == 200
    assert len(r_thumb.content) < len(r.content)

    # Path traversal protection
    r_bad = client.get("/api/catalogue/image?rel=../../etc/passwd")
    assert r_bad.status_code in (400, 404)

    # Non-existent image
    r_missing = client.get("/api/catalogue/image?rel=does_not_exist.jpg")
    assert r_missing.status_code == 404


def test_api_commercial_matches_endpoint(tmp_path):
    import server.app as app_mod
    cache_dir = app_mod.CFG.paths.cache_dir
    vid = "testmatchvid123"
    vdir = cache_dir / vid
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "upload.json").write_text(json.dumps({"name": "test.mp4", "path": str(tmp_path / "test.mp4")}))
    (vdir / "info.json").write_text(json.dumps({
        "path": str(tmp_path / "test.mp4"),
        "duration": 10.0,
        "fps": 25.0,
        "width": 640,
        "height": 360,
        "video_codec": "h264",
        "sar": "1:1",
        "dar": "16:9",
    }))
    (vdir / "result.json").write_text(json.dumps({
        "video": "test.mp4",
        "duration": 10.0,
        "scene_count": 1,
        "scenes": [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": 10.0, "shot_start": 0, "shot_end": 0}],
        "debug": {"shots": [{"shot_id": 0, "start_frame": 0, "end_frame": 100, "start_seconds": 0.0, "end_seconds": 10.0}],
                  "feature_key": "feat123", "timings": {}}
    }))
    comm_dir = vdir / "commercial"
    comm_dir.mkdir(exist_ok=True)
    (comm_dir / "matches.json").write_text(json.dumps({
        "cand1": [{
            "product_id": "p1", "brand": "IN YOUR SHOE", "name": "Item 1",
            "category": "Fashion", "product_type": "Tee", "price": 499.0,
            "compare_at_price": None, "currency": "EGP", "url": "https://example.com",
            "similarity": 0.85, "best_image_rel": "rel/path.jpg", "best_image_url": ""
        }]
    }))

    client = TestClient(app_mod.app)
    r = client.get(f"/api/videos/{vid}/commercial/matches?compute=false")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert "cand1" in data["matches"]
    assert data["matches"]["cand1"][0]["name"] == "Item 1"
    assert data["matches"]["cand1"][0]["similarity"] == 0.85
