"""Catalogue operations: brands, products, variants, reference images, search.

All functions take a `Database` and return plain dictionaries, so the web layer and the
matching code never touch ORM objects. Problems raise CatalogError with a stable `code`
(not_found | conflict | invalid) that the API maps to 404 / 409 / 400.
"""
from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from ..commercial import taxonomy as T
from .db import AVAILABILITY, IMAGE_ROLES, Brand, Database, Product, ProductImage, ProductVariant, Identification, utcnow
from .images import ImageError, ImageStore

# Catalogue products live in the object categories (venues / real estate are scene contexts).
PRODUCT_CATEGORIES = tuple(c for c in T.CATEGORIES if any(o.category == c for o in T.OBJECTS.values()))


class CatalogError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _clean(v):
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def _url(v):
    """Only http(s) links are stored: they are rendered as clickable links in the UI."""
    v = _clean(v)
    if v is None:
        return None
    from urllib.parse import urlparse

    u = urlparse(v)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise CatalogError("invalid", "links must start with http:// or https://")
    return v[:2000]


def _need(row, what: str, id_):
    if row is None:
        raise CatalogError("not_found", f"{what} {id_} does not exist")
    return row


# ---------------------------------------------------------------- serialisation

def brand_dict(b: Brand, products: int | None = None) -> dict:
    d = {"id": b.id, "name": b.name, "website": b.website, "description": b.description, "metadata": b.extra or {},
         "archived": b.archived, "created_at": b.created_at.isoformat(), "updated_at": b.updated_at.isoformat()}
    if products is not None:
        d["product_count"] = products
    return d


def image_dict(i: ProductImage) -> dict:
    return {"id": i.id, "product_id": i.product_id, "variant_id": i.variant_id, "role": i.role, "sha256": i.sha256,
            "ext": i.ext, "width": i.width, "height": i.height, "size_bytes": i.size_bytes,
            "original_filename": i.original_filename, "position": i.position,
            "url": f"/api/catalog/images/{i.sha256}.{i.ext}"}


def variant_dict(v: ProductVariant) -> dict:
    return {"id": v.id, "product_id": v.product_id, "name": v.name, "color": v.color, "size": v.size, "sku": v.sku,
            "external_id": v.external_id, "url": v.url, "price": float(v.price) if v.price is not None else None,
            "currency": v.currency, "availability": v.availability, "metadata": v.extra or {}}


def product_dict(p: Product, full: bool = True) -> dict:
    obj = T.OBJECTS.get(p.object_type) if p.object_type else None
    d = {"id": p.id, "brand": {"id": p.brand.id, "name": p.brand.name}, "name": p.name, "category": p.category,
         "category_name": T.CATEGORIES[p.category].name if p.category in T.CATEGORIES else p.category,
         "object_type": p.object_type, "object_label": obj.label if obj else None, "sku": p.sku,
         "external_id": p.external_id, "url": p.url, "price": float(p.price) if p.price is not None else None,
         "currency": p.currency, "availability": p.availability, "archived": p.archived,
         "image_count": len(p.images), "variant_count": len(p.variants),
         "primary_image": image_dict(p.images[0]) if p.images else None,
         "updated_at": p.updated_at.isoformat()}
    if full:
        d.update({"description": p.description, "metadata": p.extra or {}, "created_at": p.created_at.isoformat(),
                  "images": [image_dict(i) for i in p.images], "variants": [variant_dict(v) for v in p.variants]})
    return d


# ---------------------------------------------------------------- brands

def create_brand(db: Database, name: str, website: str | None = None, description: str | None = None,
                 metadata: dict | None = None) -> dict:
    name = _clean(name)
    if not name:
        raise CatalogError("invalid", "a brand needs a name")
    with db.session() as s:
        if s.scalar(select(Brand).where(func.lower(Brand.name) == name.lower())):
            raise CatalogError("conflict", f"a brand named '{name}' already exists")
        b = Brand(name=name, website=_url(website), description=_clean(description), extra=metadata or {})
        s.add(b)
        s.flush()
        return brand_dict(b, 0)


def update_brand(db: Database, brand_id: int, **fields) -> dict:
    with db.session() as s:
        b = _need(s.get(Brand, brand_id), "brand", brand_id)
        if "name" in fields:
            name = _clean(fields["name"])
            if not name:
                raise CatalogError("invalid", "a brand needs a name")
            clash = s.scalar(select(Brand).where(func.lower(Brand.name) == name.lower(), Brand.id != brand_id))
            if clash:
                raise CatalogError("conflict", f"a brand named '{name}' already exists")
            b.name = name
        if "website" in fields:
            b.website = _url(fields["website"])
        if "description" in fields:
            b.description = _clean(fields["description"])
        if "metadata" in fields:
            b.extra = fields["metadata"] or {}
        if "archived" in fields:
            b.archived = bool(fields["archived"])
        b.updated_at = utcnow()
        s.flush()
        return brand_dict(b)


def list_brands(db: Database, include_archived: bool = False, q: str | None = None,
                compatible_with: str | None = None, limit: int | None = None) -> list[dict]:
    """Brands with their product counts. `q` searches the name; `compatible_with` (a detected
    object type) counts only products that kind of object could be and lists those brands first."""
    with db.session() as s:
        on = (Product.brand_id == Brand.id) & (Product.archived.is_(False))
        if compatible_with:
            on = on & compatible_filter(compatible_with)
        n = func.count(Product.id)
        stmt = select(Brand, n).outerjoin(Product, on).group_by(Brand.id)
        stmt = stmt.order_by(n.desc(), func.lower(Brand.name)) if compatible_with else stmt.order_by(func.lower(Brand.name))
        if not include_archived:
            stmt = stmt.where(Brand.archived.is_(False))
        for word in (q or "").lower().split()[:6]:
            stmt = stmt.where(func.lower(Brand.name).like(f"%{word}%"))
        if limit:
            stmt = stmt.limit(max(1, min(int(limit), 500)))
        return [brand_dict(b, c) for b, c in s.execute(stmt)]


# ---------------------------------------------------------------- products

def _validate_product(fields: dict, partial: bool) -> dict:
    out = {}
    if "name" in fields or not partial:
        out["name"] = _clean(fields.get("name"))
        if not out["name"]:
            raise CatalogError("invalid", "a product needs a name")
    if "category" in fields or not partial:
        cat = _clean(fields.get("category"))
        if cat not in PRODUCT_CATEGORIES:
            raise CatalogError("invalid", f"category must be one of: {', '.join(PRODUCT_CATEGORIES)}")
        out["category"] = cat
    if "object_type" in fields:
        ot = _clean(fields.get("object_type"))
        if ot is not None and ot not in T.OBJECTS:
            raise CatalogError("invalid", f"unknown object type '{ot}'")
        out["object_type"] = ot
    for k in ("description", "sku", "external_id", "currency"):
        if k in fields:
            out[k] = _clean(fields[k])
    if "url" in fields:
        out["url"] = _url(fields["url"])
    if out.get("currency"):
        out["currency"] = out["currency"].upper()[:3]
    if "price" in fields:
        price = fields["price"]
        if price in ("", None):
            out["price"] = None
        else:
            try:
                out["price"] = round(float(price), 2)
            except (TypeError, ValueError):
                raise CatalogError("invalid", "price must be a number")
            if out["price"] < 0:
                raise CatalogError("invalid", "price cannot be negative")
    if "availability" in fields:
        av = _clean(fields["availability"]) or "unknown"
        if av not in AVAILABILITY:
            raise CatalogError("invalid", f"availability must be one of: {', '.join(AVAILABILITY)}")
        out["availability"] = av
    if "metadata" in fields:
        if fields["metadata"] is not None and not isinstance(fields["metadata"], dict):
            raise CatalogError("invalid", "metadata must be an object")
        out["extra"] = fields["metadata"] or {}
    if "archived" in fields:
        out["archived"] = bool(fields["archived"])
    return out


def _check_type_in_category(category: str, object_type: str | None) -> None:
    if object_type and T.OBJECTS[object_type].category != category:
        raise CatalogError("invalid", f"'{T.OBJECTS[object_type].label}' belongs to category "
                                      f"'{T.OBJECTS[object_type].category}', not '{category}'")


def _integrity(e: IntegrityError, fields: dict) -> CatalogError:
    msg = str(e.orig).lower()
    if "sku" in msg:
        return CatalogError("conflict", f"this brand already has a product with SKU '{fields.get('sku')}'")
    if "name" in msg or "unique" in msg:
        return CatalogError("conflict", f"this brand already has a product named '{fields.get('name')}'")
    return CatalogError("invalid", "the product could not be saved")


def create_product(db: Database, brand_id: int, **fields) -> dict:
    data = _validate_product(fields, partial=False)
    _check_type_in_category(data["category"], data.get("object_type"))
    try:
        with db.session() as s:
            _need(s.get(Brand, brand_id), "brand", brand_id)
            p = Product(brand_id=brand_id, **data)
            s.add(p)
            s.flush()
            return product_dict(_load(s, p.id))
    except IntegrityError as e:
        raise _integrity(e, data) from e


def update_product(db: Database, product_id: int, **fields) -> dict:
    data = _validate_product(fields, partial=True)
    try:
        with db.session() as s:
            p = _need(_load(s, product_id), "product", product_id)
            if "brand_id" in fields and fields["brand_id"] is not None:
                _need(s.get(Brand, int(fields["brand_id"])), "brand", fields["brand_id"])
                p.brand_id = int(fields["brand_id"])
            for k, v in data.items():
                setattr(p, k, v)
            _check_type_in_category(p.category, p.object_type)
            p.updated_at = utcnow()
            s.flush()
            s.refresh(p)
            return product_dict(_load(s, product_id))
    except IntegrityError as e:
        raise _integrity(e, {**fields}) from e


def _load(s, product_id: int) -> Product | None:
    return s.scalar(select(Product).where(Product.id == product_id).options(
        selectinload(Product.brand), selectinload(Product.images), selectinload(Product.variants)))


def get_product(db: Database, product_id: int) -> dict:
    with db.session() as s:
        return product_dict(_need(_load(s, product_id), "product", product_id))


def delete_product(db: Database, product_id: int) -> dict:
    """Remove a product. A product that a person has identified in a scene is archived instead
    of deleted, so identifications and their history keep pointing at something."""
    with db.session() as s:
        p = _need(_load(s, product_id), "product", product_id)
        used = s.scalar(select(func.count(Identification.id)).where(Identification.product_id == product_id))
        if used:
            p.archived = True
            p.updated_at = utcnow()
            return {"deleted": False, "archived": True, "reason": f"identified in {used} scene object(s)"}
        s.delete(p)
        return {"deleted": True, "archived": False}


def compatible_filter(type_id: str):
    """Products a detected object of `type_id` could be: the same or an easily-confused object
    type (sneakers / shoes), or an untyped product of the same category. This only narrows what
    a person has to look through; it never picks a product."""
    obj = T.OBJECTS.get(type_id)
    if obj is None:
        raise CatalogError("invalid", f"unknown object type '{type_id}'")
    return or_(Product.object_type.in_(list(T.compatible_types(type_id))),
               (Product.object_type.is_(None)) & (Product.category == obj.category))


def search_products(db: Database, q: str | None = None, category: str | None = None, object_type: str | None = None,
                    brand_id: int | None = None, include_archived: bool = False, limit: int = 48,
                    offset: int = 0, compatible_with: str | None = None) -> dict:
    """Paged product search (name, SKU, external id, brand name, variant SKU). Filtering and
    paging happen in the database, so this stays fast with thousands of products.
    `compatible_with` = a detected object type: only products that kind of object could be."""
    limit, offset = max(1, min(int(limit), 200)), max(0, int(offset))
    with db.session() as s:
        cond = []
        if not include_archived:
            cond.append(Product.archived.is_(False))
        if category:
            cond.append(Product.category == category)
        if object_type:
            cond.append(Product.object_type == object_type)
        if brand_id:
            cond.append(Product.brand_id == brand_id)
        if compatible_with:
            cond.append(compatible_filter(compatible_with))
        base = select(Product).join(Brand)
        for word in (q or "").lower().split()[:6]:               # every word must match somewhere
            like = f"%{word}%"
            cond.append(or_(func.lower(Product.name).like(like), func.lower(Product.sku).like(like),
                            func.lower(Product.external_id).like(like), func.lower(Brand.name).like(like),
                            Product.id.in_(select(ProductVariant.product_id).where(func.lower(ProductVariant.sku).like(like)))))
        total = s.scalar(select(func.count()).select_from(base.where(*cond).subquery()))
        rows = s.scalars(base.where(*cond).order_by(Product.updated_at.desc(), Product.id.desc())
                         .limit(limit).offset(offset)
                         .options(selectinload(Product.brand), selectinload(Product.images),
                                  selectinload(Product.variants))).all()
        return {"total": int(total or 0), "limit": limit, "offset": offset,
                "items": [product_dict(p, full=False) for p in rows]}


def category_counts(db: Database) -> dict[str, int]:
    with db.session() as s:
        return {c: n for c, n in s.execute(select(Product.category, func.count(Product.id))
                                           .where(Product.archived.is_(False)).group_by(Product.category))}


# ---------------------------------------------------------------- variants

def _variant_fields(fields: dict) -> dict:
    out = {}
    for k in ("name", "color", "size", "sku", "external_id", "currency"):
        if k in fields:
            out[k] = _clean(fields[k])
    if "url" in fields:
        out["url"] = _url(fields["url"])
    if "price" in fields:
        try:
            out["price"] = None if fields["price"] in ("", None) else round(float(fields["price"]), 2)
        except (TypeError, ValueError):
            raise CatalogError("invalid", "price must be a number")
    if "availability" in fields:
        av = _clean(fields["availability"]) or "unknown"
        if av not in AVAILABILITY:
            raise CatalogError("invalid", f"availability must be one of: {', '.join(AVAILABILITY)}")
        out["availability"] = av
    if "metadata" in fields:
        out["extra"] = fields["metadata"] or {}
    return out


def add_variant(db: Database, product_id: int, **fields) -> dict:
    data = _variant_fields(fields)
    if not data.get("name"):
        raise CatalogError("invalid", "a variant needs a name (e.g. 'White / 42')")
    try:
        with db.session() as s:
            _need(s.get(Product, product_id), "product", product_id)
            v = ProductVariant(product_id=product_id, **data)
            s.add(v)
            s.flush()
            return variant_dict(v)
    except IntegrityError as e:
        raise CatalogError("conflict", f"this product already has a variant named '{data['name']}'") from e


def update_variant(db: Database, variant_id: int, **fields) -> dict:
    data = _variant_fields(fields)
    if "name" in data and not data["name"]:
        raise CatalogError("invalid", "a variant needs a name")
    try:
        with db.session() as s:
            v = _need(s.get(ProductVariant, variant_id), "variant", variant_id)
            for k, val in data.items():
                setattr(v, k, val)
            s.flush()
            return variant_dict(v)
    except IntegrityError as e:
        raise CatalogError("conflict", "this product already has a variant with that name") from e


def delete_variant(db: Database, variant_id: int) -> None:
    with db.session() as s:
        if s.scalar(select(func.count(Identification.id)).where(Identification.variant_id == variant_id)):
            raise CatalogError("conflict", "this variant is used by an identification; change that identification first")
        s.delete(_need(s.get(ProductVariant, variant_id), "variant", variant_id))


# ---------------------------------------------------------------- images

def add_image(db: Database, store: ImageStore, product_id: int, data: bytes, role: str = "front",
              filename: str | None = None, variant_id: int | None = None) -> dict:
    if role not in IMAGE_ROLES:
        raise CatalogError("invalid", f"image role must be one of: {', '.join(IMAGE_ROLES)}")
    try:
        meta = store.save(data)
    except ImageError as e:
        raise CatalogError("invalid", str(e)) from e
    try:
        with db.session() as s:
            p = _need(s.get(Product, product_id), "product", product_id)
            if variant_id is not None:
                v = _need(s.get(ProductVariant, variant_id), "variant", variant_id)
                if v.product_id != product_id:
                    raise CatalogError("invalid", "that variant belongs to another product")
            pos = s.scalar(select(func.coalesce(func.max(ProductImage.position), -1))
                           .where(ProductImage.product_id == product_id)) + 1
            img = ProductImage(product_id=product_id, variant_id=variant_id, role=role,
                               original_filename=(filename or "")[:300] or None, position=pos, **meta)
            s.add(img)
            p.updated_at = utcnow()
            s.flush()
            return image_dict(img)
    except IntegrityError as e:
        raise CatalogError("conflict", "this exact image is already attached to the product") from e


def update_image(db: Database, image_id: int, role: str | None = None, position: int | None = None,
                 variant_id: int | None = None) -> dict:
    with db.session() as s:
        img = _need(s.get(ProductImage, image_id), "image", image_id)
        if role is not None:
            if role not in IMAGE_ROLES:
                raise CatalogError("invalid", f"image role must be one of: {', '.join(IMAGE_ROLES)}")
            img.role = role
        if position is not None:
            img.position = int(position)
        if variant_id is not None:
            img.variant_id = variant_id or None
        s.flush()
        return image_dict(img)


def delete_image(db: Database, image_id: int) -> None:
    """Detach an image from its product. The file stays in the content-addressed store (another
    product may use the same content, and cached embeddings stay valid)."""
    with db.session() as s:
        s.delete(_need(s.get(ProductImage, image_id), "image", image_id))


def stats(db: Database) -> dict:
    with db.session() as s:
        return {"brands": s.scalar(select(func.count(Brand.id)).where(Brand.archived.is_(False))),
                "products": s.scalar(select(func.count(Product.id)).where(Product.archived.is_(False))),
                "images": s.scalar(select(func.count(ProductImage.id))),
                "archived_products": s.scalar(select(func.count(Product.id)).where(Product.archived.is_(True)))}
