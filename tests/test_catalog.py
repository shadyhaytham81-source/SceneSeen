"""Phase 2B catalogue: persistence, CRUD, images, search. SQLite in a temp folder."""
import numpy as np
import pytest

from sceneseen.catalog import service as S
from sceneseen.catalog.db import Database, ImageEmbedding
from sceneseen.catalog.images import ImageError, ImageStore
from sceneseen.catalog import identification as I

from .fakes import jpeg, pattern


@pytest.fixture
def cat(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'catalog.db'}")
    yield db, ImageStore(tmp_path / "images")
    db.dispose()


def product(db, brand="Acme", name="Phone X", **kw):
    brands = {b["name"]: b["id"] for b in S.list_brands(db, include_archived=True)}
    bid = brands.get(brand) or S.create_brand(db, brand)["id"]
    kw.setdefault("category", "electronics")
    return S.create_product(db, bid, name=name, **kw)


def test_brand_crud_and_duplicates(cat):
    db, _ = cat
    b = S.create_brand(db, "  Acme  ", website="https://acme.example")
    assert b["name"] == "Acme" and b["product_count"] == 0
    with pytest.raises(S.CatalogError) as e:
        S.create_brand(db, "acme")                       # case-insensitive duplicate
    assert e.value.code == "conflict"
    with pytest.raises(S.CatalogError):
        S.create_brand(db, "   ")
    assert S.update_brand(db, b["id"], name="Acme Co", archived=True)["archived"] is True
    assert S.list_brands(db) == [] and len(S.list_brands(db, include_archived=True)) == 1
    with pytest.raises(S.CatalogError) as e:
        S.update_brand(db, 999, name="x")
    assert e.value.code == "not_found"


def test_product_crud(cat):
    db, _ = cat
    p = product(db, object_type="smartphone", sku="PX-1", url="https://acme.example/px", price=19999.5, currency="EGP",
                external_id="ext-9", metadata={"season": "2026"})
    assert p["brand"]["name"] == "Acme" and p["object_label"] == "Smartphone" and p["price"] == 19999.5
    assert p["metadata"] == {"season": "2026"} and p["availability"] == "unknown"
    u = S.update_product(db, p["id"], name="Phone X2", availability="in_stock", price=None)
    assert u["name"] == "Phone X2" and u["availability"] == "in_stock" and u["price"] is None and u["sku"] == "PX-1"
    assert S.get_product(db, p["id"])["name"] == "Phone X2"
    assert S.delete_product(db, p["id"]) == {"deleted": True, "archived": False}
    with pytest.raises(S.CatalogError) as e:
        S.get_product(db, p["id"])
    assert e.value.code == "not_found"


def test_product_validation(cat):
    db, _ = cat
    bid = S.create_brand(db, "Acme")["id"]
    for bad in ({"name": "", "category": "electronics"}, {"name": "A", "category": "spaceships"},
                {"name": "A", "category": "electronics", "object_type": "sofa"},        # type of another category
                {"name": "A", "category": "electronics", "price": -5},
                {"name": "A", "category": "electronics", "url": "javascript:alert(1)"},
                {"name": "A", "category": "electronics", "availability": "maybe"}):
        with pytest.raises(S.CatalogError) as e:
            S.create_product(db, bid, **bad)
        assert e.value.code == "invalid", bad
    with pytest.raises(S.CatalogError) as e:
        S.create_product(db, 12345, name="A", category="electronics")
    assert e.value.code == "not_found"


def test_duplicate_products_are_rejected_per_brand(cat):
    db, _ = cat
    product(db, name="Phone X", sku="S1")
    with pytest.raises(S.CatalogError) as e:
        product(db, name="Phone X")
    assert e.value.code == "conflict"
    with pytest.raises(S.CatalogError) as e:
        product(db, name="Phone Y", sku="S1")
    assert e.value.code == "conflict"
    assert product(db, brand="Other", name="Phone X", sku="S1")["id"]      # another brand may reuse name and SKU
    assert S.stats(db)["products"] == 2


def test_arabic_and_emoji_names_round_trip_and_search(cat):
    db, _ = cat
    p = product(db, brand="القاهرة للساعات", name="ساعة يد كلاسيك ⌚", category="accessories", object_type="watch",
                description="ساعة رجالي — جلد طبيعي")
    got = S.get_product(db, p["id"])
    assert got["name"] == "ساعة يد كلاسيك ⌚" and got["brand"]["name"] == "القاهرة للساعات"
    assert got["description"] == "ساعة رجالي — جلد طبيعي"
    assert [x["id"] for x in S.search_products(db, q="ساعة")["items"]] == [p["id"]]
    assert [x["id"] for x in S.search_products(db, q="القاهرة")["items"]] == [p["id"]]      # by brand name
    assert S.search_products(db, q="حذاء")["total"] == 0


def test_multiple_images_roles_and_content_addressing(cat):
    db, store = cat
    p = product(db)
    a = S.add_image(db, store, p["id"], jpeg(pattern(1)), "front", "front.jpg")
    b = S.add_image(db, store, p["id"], jpeg(pattern(2), "PNG"), "back", "back.png")
    c = S.add_image(db, store, p["id"], jpeg(pattern(3)), "detail")
    assert (a["ext"], b["ext"]) == ("jpg", "png") and a["width"] == 240
    got = S.get_product(db, p["id"])
    assert [i["role"] for i in got["images"]] == ["front", "back", "detail"] and got["image_count"] == 3
    assert got["primary_image"]["id"] == a["id"] and store.path(a["sha256"], "jpg").exists()
    with pytest.raises(S.CatalogError) as e:                      # the very same picture twice on one product
        S.add_image(db, store, p["id"], jpeg(pattern(1)), "side")
    assert e.value.code == "conflict"
    other = product(db, name="Phone Y")
    same = S.add_image(db, store, other["id"], jpeg(pattern(1)))           # same content, other product: one file
    assert same["sha256"] == a["sha256"]
    assert S.update_image(db, c["id"], role="lifestyle")["role"] == "lifestyle"
    with pytest.raises(S.CatalogError):
        S.update_image(db, c["id"], role="selfie")
    S.delete_image(db, b["id"])
    assert S.get_product(db, p["id"])["image_count"] == 2
    assert store.path(b["sha256"], "png").exists()                # file kept: content-addressed, embeddings stay valid


@pytest.mark.parametrize("data,why", [(b"", "empty"), (b"this is not an image", "garbage"),
                                      (jpeg(pattern(1))[:200], "truncated"), (jpeg(np.zeros((8, 8, 3), np.uint8)), "tiny")])
def test_bad_images_are_rejected_cleanly(cat, data, why):
    db, store = cat
    p = product(db)
    with pytest.raises(S.CatalogError) as e:
        S.add_image(db, store, p["id"], data)
    assert e.value.code == "invalid", why
    assert S.get_product(db, p["id"])["image_count"] == 0
    with pytest.raises(ImageError):
        store.save(data)


def test_image_for_unknown_product(cat):
    db, store = cat
    with pytest.raises(S.CatalogError) as e:
        S.add_image(db, store, 4242, jpeg(pattern(1)))
    assert e.value.code == "not_found"


def test_variants(cat):
    db, _ = cat
    p = product(db, name="Runner", category="fashion", object_type="sneakers")
    v = S.add_variant(db, p["id"], name="White / 42", color="white", size="42", sku="R-W-42", price=3200)
    assert v["color"] == "white" and S.get_product(db, p["id"])["variant_count"] == 1
    assert S.update_variant(db, v["id"], size="43")["size"] == "43"
    S.delete_variant(db, v["id"])
    assert S.get_product(db, p["id"])["variants"] == []
    with pytest.raises(S.CatalogError):
        S.add_variant(db, 999, name="x")


def test_search_filter_and_paging_on_a_larger_catalogue(cat):
    db, _ = cat
    bid = S.create_brand(db, "Bulk")["id"]
    from sceneseen.catalog.db import Product

    with db.session() as s:
        s.add_all([Product(brand_id=bid, name=f"Item {i:04d}", category="fashion" if i % 2 else "electronics",
                           object_type="sneakers" if i % 2 else "laptop", sku=f"SKU{i:04d}") for i in range(1200)])
    r = S.search_products(db, limit=50)
    assert r["total"] == 1200 and len(r["items"]) == 50
    page2 = S.search_products(db, limit=50, offset=50)
    assert not {x["id"] for x in r["items"]} & {x["id"] for x in page2["items"]}
    assert S.search_products(db, category="fashion")["total"] == 600
    assert S.search_products(db, object_type="laptop")["total"] == 600
    assert S.search_products(db, q="SKU0777")["items"][0]["name"] == "Item 0777"
    assert S.search_products(db, q="item 000", category="electronics")["total"] == 6      # 0000..0008 even + 1000
    assert S.search_products(db, limit=10 ** 6)["limit"] == 200            # page size is capped
    assert S.category_counts(db) == {"electronics": 600, "fashion": 600}
    S.update_product(db, r["items"][0]["id"], archived=True)
    assert S.search_products(db)["total"] == 1199 and S.search_products(db, include_archived=True)["total"] == 1200


def test_identified_product_is_archived_not_deleted(cat):
    db, _ = cat
    p = product(db)
    v = S.add_variant(db, p["id"], name="Black")
    obj = {"key": "abc123", "type_id": "smartphone", "label": "Smartphone", "scene_id": 1}
    I.identify(db, "vid1", obj, "shady", product_id=p["id"], variant_id=v["id"])
    with pytest.raises(S.CatalogError) as e:
        S.delete_variant(db, v["id"])                       # a variant in use cannot vanish
    assert e.value.code == "conflict"
    r = S.delete_product(db, p["id"])
    assert r["deleted"] is False and r["archived"] is True
    assert S.get_product(db, p["id"])["archived"] is True
    assert I.for_video(db, "vid1")["abc123"]["product"]["archived"] is True      # the identification still resolves


def test_compatible_filter_and_sku_search(cat):
    """Once the generic object is known, only products that object could be are offered."""
    db, _ = cat
    sneaker = product(db, brand="Nike", name="Air Force 1", category="fashion", object_type="sneakers", sku="AF1-WHT")
    shoe = product(db, brand="Clarks", name="Desert Boot", category="fashion", object_type="shoes", sku="CL-1")
    untyped = product(db, brand="Nike", name="Mystery fashion item", category="fashion")
    shirt = product(db, brand="Nike", name="Dri-FIT Tee", category="fashion", object_type="shirt")
    tv = product(db, brand="Samsung", name="Frame TV", category="electronics", object_type="television")
    phone = product(db, brand="Samsung", name="Galaxy S", category="electronics", object_type="smartphone")
    ids = lambda r: {x["id"] for x in r["items"]}     # noqa: E731
    got = ids(S.search_products(db, compatible_with="sneakers"))
    assert got == {sneaker["id"], shoe["id"], untyped["id"]}        # footwear group + untyped fashion; no TV, phone, shirt
    assert ids(S.search_products(db, compatible_with="television")) == {tv["id"]}
    assert ids(S.search_products(db, compatible_with="sneakers", brand_id=sneaker["brand"]["id"])) == {sneaker["id"], untyped["id"]}
    assert ids(S.search_products(db, q="af1-wht")) == {sneaker["id"]}                 # SKU search
    assert ids(S.search_products(db, q="nike force")) == {sneaker["id"]}              # brand + name words
    assert ids(S.search_products(db, q="galaxy", compatible_with="sneakers")) == set()
    v = S.add_variant(db, shoe["id"], name="Sand / 43", sku="CL-1-SND-43")
    assert ids(S.search_products(db, q="snd-43")) == {shoe["id"]} and v["sku"]        # variant SKU finds its product
    with pytest.raises(S.CatalogError):
        S.search_products(db, compatible_with="not_a_type")
    brands = S.list_brands(db, compatible_with="sneakers")
    assert [(b["name"], b["product_count"]) for b in brands] == [("Nike", 2), ("Clarks", 1), ("Samsung", 0)]
    assert [b["name"] for b in S.list_brands(db, q="sam")] == ["Samsung"]
    assert phone["id"] and shirt["id"]


def test_database_survives_reopen_and_reports_schema(tmp_path):
    url = f"sqlite:///{tmp_path / 'c.db'}"
    db = Database(url)
    p = product(db, name="Persist")
    db.dispose()
    db2 = Database(url)
    assert S.get_product(db2, p["id"])["name"] == "Persist"
    with db2.session() as s:
        assert s.query(ImageEmbedding).count() == 0
    db2.dispose()
