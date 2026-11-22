"""App backend stubs (backend/): routes, roles, labelling rules and catalogue import.

Uses a fake detector and a fresh in-memory store per test; no models are downloaded.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import deps
from backend.main import app
from backend.schemas import Box, Brand, DetectedItem, Episode, Show
from backend.services.embedder import Embedder
from backend.services.scan_service import ScanService
from backend.store import Store

IMAGE = b"fake-screen-photo"
ADMIN = {"X-User-Id": "admin1", "X-User-Role": "admin"}
VIEWER = {"X-User-Id": "viewer1", "X-User-Role": "viewer"}


class OneDressDetector:
    def detect(self, image):
        return [DetectedItem(category="dress", colour="red", confidence=0.9, box=Box(x=0.1, y=0.1, w=0.5, h=0.8))]


class FixedEmbedder(Embedder):
    """Scan frames and the 'match' product image share one vector; everything else is random."""
    def embed(self, image):
        return super().embed(b"same" if image in (IMAGE, b"https://img/match.jpg") else image)


@pytest.fixture
def env():
    store = Store()
    svc = ScanService(store, detector=OneDressDetector(), embedder=FixedEmbedder())
    app.dependency_overrides[deps.get_store] = lambda: store
    app.dependency_overrides[deps.get_scan_service] = lambda: svc
    app.dependency_overrides[deps.get_embedder] = lambda: svc.embedder
    from backend.services.analytics import AnalyticsService
    from backend.services.recommender import Recommender
    app.dependency_overrides[deps.get_analytics] = lambda: AnalyticsService(store)
    app.dependency_overrides[deps.get_recommender] = lambda: Recommender(store)
    brand = Brand(name="Test Brand")
    store.brands[brand.id] = brand
    show = Show(title="Test Show")
    ep = Episode(show_id=show.id, season=1, number=1)
    store.shows[show.id], store.episodes[ep.id] = show, ep
    yield TestClient(app), store, brand, ep
    app.dependency_overrides.clear()


TEMPLATE = (b"title,category,price,product_url,image_url,colour\n"
            b"Red dress,dress,1200,https://shop/red,https://img/match.jpg,red\n"
            b"Blue shirt,shirt,500,https://shop/blue,https://img/other.jpg,blue\n"
            b",dress,100,https://shop/x,,\n")


def _import(client, brand):
    return client.post(f"/brands/{brand.id}/catalogue", headers=ADMIN, data={"format": "template"},
                       files={"file": ("c.csv", TEMPLATE, "text/csv")})


def _scan(client, headers=None):
    r = client.post("/scans", files={"image": ("s.jpg", IMAGE, "image/jpeg")}, headers=headers or {})
    assert r.status_code == 201
    return r.json()["scan_id"]


def test_health(env):
    client, *_ = env
    assert client.get("/health").json()["status"] == "ok"


def test_import_reports_bad_rows(env):
    client, store, brand, _ = env
    r = _import(client, brand)
    assert r.status_code == 201
    assert r.json()["imported"] == 2 and r.json()["skipped_rows"] == [3]


def test_viewer_cannot_import(env):
    client, _, brand, _ = env
    r = client.post(f"/brands/{brand.id}/catalogue", headers=VIEWER, files={"file": ("c.csv", TEMPLATE)})
    assert r.status_code == 403


def test_scan_returns_similar_then_exact_after_approval(env):
    client, store, brand, ep = env
    _import(client, brand)
    sid = _scan(client)
    items = client.get(f"/scans/{sid}/results").json()["items"]
    assert len(items) == 1
    [m] = items[0]["matches"]                       # shirt is filtered out by category
    assert m["product"]["title"] == "Red dress" and m["label"] == "Similar"

    product_id = m["product"]["id"]
    stylist = {"X-User-Id": "sty1", "X-User-Role": "stylist"}
    v = client.post("/verified-items", headers=stylist, json={"episode_id": ep.id, "product_id": product_id}).json()
    assert v["status"] == "pending"
    assert client.post(f"/verified-items/{v['id']}/approve", headers=stylist).status_code == 403
    assert client.post(f"/verified-items/{v['id']}/approve", headers=ADMIN).json()["status"] == "approved"

    client.patch(f"/scans/{sid}", json={"episode_id": ep.id})
    m = client.get(f"/scans/{sid}/results").json()["items"][0]["matches"][0]
    assert m["label"] == "Exact"


def test_no_close_match_message(env):
    client, *_ = env                                 # no catalogue imported
    sid = _scan(client)
    item = client.get(f"/scans/{sid}/results").json()["items"][0]
    assert item["matches"] == [] and item["message"] == "no close match yet"


def test_episode_suggested_from_earlier_tagged_scan(env):
    client, _, _, ep = env
    first = _scan(client)
    client.patch(f"/scans/{first}", json={"episode_id": ep.id})
    r = client.post("/scans", files={"image": ("s.jpg", IMAGE)})
    assert r.json()["suggested_episode"]["id"] == ep.id


def test_redirect_records_click_and_report_aggregates(env):
    client, store, brand, _ = env
    _import(client, brand)
    sid = _scan(client, VIEWER)
    link = client.get(f"/scans/{sid}/results", headers=VIEWER).json()["items"][0]["matches"][0]["link"]
    r = client.get(link, headers=VIEWER, follow_redirects=False)
    assert r.status_code == 307 and "utm_source=sceneseen" in r.headers["location"]

    partner = {"X-User-Id": "bp1", "X-User-Role": "brand_partner", "X-Brand-Id": brand.id}
    rep = client.get(f"/brands/{brand.id}/report", headers=partner).json()
    assert rep["clicks"] == 1 and rep["views"] == 1
    assert "user_id" not in str(rep)


def test_other_users_scan_is_private(env):
    client, *_ = env
    sid = _scan(client, VIEWER)
    other = {"X-User-Id": "someone-else", "X-User-Role": "viewer"}
    assert client.get(f"/scans/{sid}/results", headers=other).status_code == 403


def test_recommendations_need_login(env):
    client, _, brand, _ = env
    _import(client, brand)
    assert client.get("/recommendations").status_code == 401
    assert len(client.get("/recommendations", headers=VIEWER).json()) == 2


def test_shopify_rows_and_variants():
    from backend.services.catalogue_importer import ShopifyImporter
    csv = (b"Handle,Title,Type,Variant Price,Image Src\n"
           b"red-dress,Red Dress,Dress,1500.00,https://img/1.jpg\n"
           b"red-dress,,,1500.00,https://img/2.jpg\n")
    store = Store()
    rep = ShopifyImporter(store, Embedder(), "https://brand.com/").run(csv, "b1")
    assert rep.imported == 1
    [p] = store.products.values()
    assert p.product_url == "https://brand.com/products/red-dress" and p.category == "dress"


def test_embedder_is_unit_length():
    v = Embedder().embed(b"x")
    assert v.shape == (512,) and abs(float(np.linalg.norm(v)) - 1) < 1e-5
