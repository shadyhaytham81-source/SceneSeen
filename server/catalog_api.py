"""Catalogue API (Phase 2B): brands, products, variants, product images.

Built as a router so the catalogue stays independent from the video routes: if the database
cannot be opened, these routes answer 503 and everything else keeps working.
"""
from __future__ import annotations

import io
from typing import Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from sceneseen.catalog import service as S
from sceneseen.catalog.db import AVAILABILITY, IMAGE_ROLES
from sceneseen.commercial import taxonomy as T

STATUS = {"not_found": 404, "conflict": 409, "invalid": 400}


class BrandBody(BaseModel):
    name: str | None = None
    website: str | None = None
    description: str | None = None
    metadata: dict | None = None
    archived: bool | None = None


class ProductBody(BaseModel):
    brand_id: int | None = None
    name: str | None = None
    category: str | None = None
    object_type: str | None = None
    description: str | None = None
    sku: str | None = None
    external_id: str | None = None
    url: str | None = None
    price: float | None = None
    currency: str | None = None
    availability: str | None = None
    metadata: dict | None = None
    archived: bool | None = None


class VariantBody(BaseModel):
    name: str | None = None
    color: str | None = None
    size: str | None = None
    sku: str | None = None
    external_id: str | None = None
    url: str | None = None
    price: float | None = None
    currency: str | None = None
    availability: str | None = None
    metadata: dict | None = None


class ImageBody(BaseModel):
    role: str | None = None
    position: int | None = None
    variant_id: int | None = None


def build_router(get_db: Callable, get_store: Callable) -> APIRouter:
    r = APIRouter(prefix="/api/catalog")

    def call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except S.CatalogError as e:
            raise HTTPException(STATUS.get(e.code, 400), e.message)

    def sent(body: BaseModel) -> dict:
        return body.model_dump(exclude_unset=True)

    @r.get("/meta")
    def meta():
        """Everything the catalogue UI needs to build its forms and filters."""
        db = get_db()
        counts = S.category_counts(db)
        return {
            "categories": [{"id": c, "name": T.CATEGORIES[c].name, "icon": T.CATEGORIES[c].icon, "products": counts.get(c, 0),
                            "object_types": [{"id": o.id, "label": o.label} for o in T.OBJECTS.values() if o.category == c]}
                           for c in S.PRODUCT_CATEGORIES],
            "image_roles": list(IMAGE_ROLES), "availability": list(AVAILABILITY), "stats": S.stats(db)}

    # ---- brands
    @r.get("/brands")
    def brands(archived: bool = False):
        return {"brands": S.list_brands(get_db(), include_archived=archived)}

    @r.post("/brands")
    def create_brand(body: BrandBody):
        return call(S.create_brand, get_db(), body.name or "", body.website, body.description, body.metadata)

    @r.patch("/brands/{brand_id}")
    def update_brand(brand_id: int, body: BrandBody):
        return call(S.update_brand, get_db(), brand_id, **sent(body))

    # ---- products
    @r.get("/products")
    def products(q: str = "", category: str = "", object_type: str = "", brand_id: int | None = None,
                 archived: bool = False, limit: int = 48, offset: int = 0):
        return S.search_products(get_db(), q or None, category or None, object_type or None, brand_id, archived,
                                 limit, offset)

    @r.post("/products")
    def create_product(body: ProductBody):
        f = sent(body)
        brand_id = f.pop("brand_id", None)
        if brand_id is None:
            raise HTTPException(400, "a product needs a brand")
        f.pop("archived", None)
        return call(S.create_product, get_db(), brand_id, **f)

    @r.get("/products/{product_id}")
    def product(product_id: int):
        return call(S.get_product, get_db(), product_id)

    @r.patch("/products/{product_id}")
    def update_product(product_id: int, body: ProductBody):
        return call(S.update_product, get_db(), product_id, **sent(body))

    @r.delete("/products/{product_id}")
    def delete_product(product_id: int):
        return call(S.delete_product, get_db(), product_id)

    # ---- variants
    @r.post("/products/{product_id}/variants")
    def add_variant(product_id: int, body: VariantBody):
        return call(S.add_variant, get_db(), product_id, **sent(body))

    @r.patch("/variants/{variant_id}")
    def update_variant(variant_id: int, body: VariantBody):
        return call(S.update_variant, get_db(), variant_id, **sent(body))

    @r.delete("/variants/{variant_id}")
    def delete_variant(variant_id: int):
        call(S.delete_variant, get_db(), variant_id)
        return {"deleted": True}

    # ---- images
    @r.post("/products/{product_id}/images")
    async def add_images(product_id: int, files: list[UploadFile] = File(...), role: str = Form("front"),
                         variant_id: int | None = Form(None)):
        """Upload one or more images. Each file is judged on its own: a bad file is reported
        and the good ones are still saved."""
        db, store = get_db(), get_store()
        call(S.get_product, db, product_id)
        saved, failed = [], []
        for f in files:
            data = await f.read()
            try:
                saved.append(S.add_image(db, store, product_id, data, role, f.filename, variant_id))
            except S.CatalogError as e:
                if e.code == "not_found":
                    raise HTTPException(404, e.message)
                failed.append({"filename": f.filename, "error": e.message})
        if failed and not saved:
            raise HTTPException(400, "; ".join(f"{x['filename']}: {x['error']}" for x in failed))
        return {"saved": saved, "failed": failed}

    @r.patch("/images/{image_id}")
    def update_image(image_id: int, body: ImageBody):
        return call(S.update_image, get_db(), image_id, **sent(body))

    @r.delete("/images/{image_id}")
    def delete_image(image_id: int):
        call(S.delete_image, get_db(), image_id)
        return {"deleted": True}

    @r.get("/images/{name}")
    def image_file(name: str, size: int = 0):
        sha, _, ext = name.partition(".")
        if len(sha) != 64 or not sha.isalnum() or ext not in ("jpg", "png", "webp"):
            raise HTTPException(400, "bad image name")
        p = get_store().path(sha, ext)
        if not p.exists():
            raise HTTPException(404, "image file is missing")
        headers = {"Cache-Control": "max-age=31536000, immutable"}     # content-addressed: never changes
        if not size:
            return FileResponse(p, headers=headers)
        from PIL import Image

        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((size, size))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
        except OSError:
            raise HTTPException(404, "image file cannot be read")
        return Response(buf.getvalue(), media_type="image/jpeg", headers=headers)

    return r
