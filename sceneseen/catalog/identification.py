"""Human product identification: a person says what a detected commercial object is.

The detector only finds GENERIC objects ("sneakers"). Nothing in SceneSeen guesses a brand, a
product, a SKU or a variant. A person links the object to the catalogue:

    unidentified         nobody has looked yet (no row)
    brand_identified     the brand is known, the exact product is not
    product_identified   a product is selected but not confirmed yet
    confirmed            a person confirmed the exact product (optionally a variant)
    unknown_product      a person looked and could not tell what it is
    no_product           generic: there is no specific product to link (a plain glass, a prop)
    not_commercial       not worth linking at all

One identification belongs to one grouped object (Phase 2A groups every occurrence of an object
in a scene), so identifying it once covers all its occurrences, and never another object that
merely has the same generic label. Every change is appended to `identification_events` with the
value before it.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .db import IDENT_STATUSES, Brand, Database, Identification, IdentificationEvent, Product, ProductVariant, utcnow

STATUS_LABELS = {
    "unidentified": "Unidentified", "brand_identified": "Brand identified", "product_identified": "Product identified",
    "confirmed": "Confirmed", "unknown_product": "Unknown product", "no_product": "No product",
    "not_commercial": "Not commercially useful"}
NO_LINK = ("unknown_product", "no_product", "not_commercial")       # statuses that carry no brand / product


class IdentificationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message      # invalid | not_found


def _image(p: Product) -> str | None:
    return f"/api/catalog/images/{p.images[0].sha256}.{p.images[0].ext}" if p.images else None


def _dict(i: Identification) -> dict:
    b, p, v = i.brand, i.product, i.variant
    return {
        "status": i.status, "status_label": STATUS_LABELS[i.status], "source": i.source,
        "brand": None if b is None else {"id": b.id, "name": b.name, "archived": b.archived},
        "product": None if p is None else {"id": p.id, "name": p.name, "sku": p.sku, "url": p.url, "archived": p.archived,
                                           "category": p.category, "object_type": p.object_type, "image": _image(p)},
        "variant": None if v is None else {"id": v.id, "name": v.name, "color": v.color, "size": v.size, "sku": v.sku,
                                           "url": v.url},
        "notes": i.notes, "identified_by": i.identified_by, "identified_at": i.identified_at.isoformat(),
        "scene_id": i.scene_id, "type_id": i.type_id, "label": i.label}


def _snapshot(i: Identification | None) -> dict | None:
    """Compact, self-contained copy for the history (names included: readable even if renamed later)."""
    if i is None:
        return None
    return {"status": i.status, "brand_id": i.brand_id, "brand": i.brand.name if i.brand else None,
            "product_id": i.product_id, "product": i.product.name if i.product else None,
            "variant_id": i.variant_id,
            "variant": (i.variant.name or i.variant.sku or f"variant {i.variant.id}") if i.variant else None,
            "notes": i.notes, "identified_by": i.identified_by, "identified_at": i.identified_at.isoformat()}


def _load(s, video_id: str, key: str) -> Identification | None:
    return s.scalar(select(Identification).where(Identification.video_id == video_id, Identification.object_key == key)
                    .options(selectinload(Identification.brand), selectinload(Identification.variant),
                             selectinload(Identification.product).selectinload(Product.images)))


def identify(db: Database, video_id: str, obj: dict, actor: str, brand_id: int | None = None,
             product_id: int | None = None, variant_id: int | None = None, status: str | None = None,
             confirm: bool = True, notes: str | None = None, video_name: str | None = None) -> dict:
    """Record what a person says the object is. `obj` is the detected object (key, type_id, label, scene_id).

    * product_id [+ variant_id]  -> "confirmed" (or "product_identified" with confirm=False);
                                    the brand always follows the product
    * brand_id only              -> "brand_identified" (the exact product stays unknown)
    * status = unknown_product | no_product | not_commercial, without brand or product

    Replacing an existing identification is recorded as a "change" with the previous value.
    """
    actor = (actor or "").strip()
    if not actor:
        raise IdentificationError("invalid", "an identification needs to say who made it")
    notes = (notes or "").strip() or None
    with db.session() as s:
        product = brand = variant = None
        if status in NO_LINK:
            if brand_id or product_id or variant_id:
                raise IdentificationError("invalid", f"'{STATUS_LABELS[status]}' cannot carry a brand or a product")
            new_status = status
        elif status not in (None, "confirmed", "product_identified", "brand_identified"):
            raise IdentificationError("invalid", f"unknown status '{status}' (use one of: {', '.join(IDENT_STATUSES)})")
        elif product_id is not None:
            product = s.get(Product, int(product_id))
            if product is None:
                raise IdentificationError("not_found", f"product {product_id} does not exist")
            if brand_id is not None and int(brand_id) != product.brand_id:
                raise IdentificationError("invalid", "that product belongs to a different brand")
            brand = s.get(Brand, product.brand_id)
            if variant_id is not None:
                variant = s.get(ProductVariant, int(variant_id))
                if variant is None:
                    raise IdentificationError("not_found", f"variant {variant_id} does not exist")
                if variant.product_id != product.id:
                    raise IdentificationError("invalid", "that variant belongs to a different product")
            if status == "brand_identified":
                raise IdentificationError("invalid", "a product was given: the status cannot be 'brand identified'")
            new_status = "product_identified" if (status == "product_identified" or (status is None and not confirm)) else "confirmed"
        elif brand_id is not None:
            if variant_id is not None:
                raise IdentificationError("invalid", "a variant needs a product")
            if status in ("confirmed", "product_identified"):
                raise IdentificationError("invalid", "confirming needs a product; with only a brand the status is 'brand identified'")
            brand = s.get(Brand, int(brand_id))
            if brand is None:
                raise IdentificationError("not_found", f"brand {brand_id} does not exist")
            new_status = "brand_identified"
        else:
            raise IdentificationError("invalid", "choose a brand or a product, or mark the object as unknown / no product / "
                                                 "not commercially useful")
        row = _load(s, video_id, obj["key"])
        before = _snapshot(row)
        if row is None:
            row = Identification(video_id=video_id, object_key=obj["key"], type_id=obj.get("type_id", ""), status=new_status,
                                 identified_by=actor)
            s.add(row)
        row.video_name = video_name
        row.scene_id, row.type_id, row.label = obj.get("scene_id"), obj.get("type_id", ""), obj.get("label")
        row.status, row.source = new_status, "human"
        row.brand, row.product, row.variant = brand, product, variant
        row.notes, row.identified_by, row.identified_at = notes, actor, utcnow()
        s.flush()
        after = _snapshot(row)
        s.add(IdentificationEvent(video_id=video_id, object_key=obj["key"], action="change" if before else "identify",
                                  actor=actor, before=before, after=after, notes=notes))
        s.flush()
        return _dict(_load(s, video_id, obj["key"]))


def clear(db: Database, video_id: str, object_key: str, actor: str, notes: str | None = None) -> bool:
    """Back to "unidentified". The removed identification stays in the history."""
    actor = (actor or "").strip()
    if not actor:
        raise IdentificationError("invalid", "removing an identification needs to say who did it")
    with db.session() as s:
        row = _load(s, video_id, object_key)
        if row is None:
            return False
        s.add(IdentificationEvent(video_id=video_id, object_key=object_key, action="clear", actor=actor,
                                  before=_snapshot(row), after=None, notes=(notes or "").strip() or None))
        s.delete(row)
        return True


def for_video(db: Database, video_id: str) -> dict[str, dict]:
    """{object key: current identification} for one video."""
    with db.session() as s:
        rows = s.scalars(select(Identification).where(Identification.video_id == video_id)
                         .options(selectinload(Identification.brand), selectinload(Identification.variant),
                                  selectinload(Identification.product).selectinload(Product.images))).all()
        return {i.object_key: _dict(i) for i in rows}


def history(db: Database, video_id: str, object_key: str | None = None, limit: int = 200) -> list[dict]:
    """Audit trail, newest first."""
    with db.session() as s:
        q = select(IdentificationEvent).where(IdentificationEvent.video_id == video_id)
        if object_key:
            q = q.where(IdentificationEvent.object_key == object_key)
        return [{"id": e.id, "object_key": e.object_key, "action": e.action, "actor": e.actor, "at": e.at.isoformat(),
                 "before": e.before, "after": e.after, "notes": e.notes}
                for e in s.scalars(q.order_by(IdentificationEvent.id.desc()).limit(limit))]


# ---------------------------------------------------------------- products per scene (final output)

def relevance_level(r: float) -> str:
    return "High" if r >= 0.70 else "Medium" if r >= 0.50 else "Low"


def scene_products(commercial: dict, identifications: dict[str, dict], include_hidden: bool = False) -> dict:
    """Per scene: context + every detected commercial object with its HUMAN identification.

    The only AI numbers are the two that describe the detection (detection_confidence,
    commercial_relevance). There is no product confidence: products come from people.
    """
    scenes, counts = [], {k: 0 for k in STATUS_LABELS}
    for sc in commercial.get("scenes", []):
        objects = []
        for c in sc["candidates"]:
            if not c["displayed"] and not include_hidden:
                continue
            ident = identifications.get(c["key"])
            status = ident["status"] if ident else "unidentified"
            counts[status] += 1
            objects.append({
                "key": c["key"], "scene_id": sc["scene_id"], "type_id": c["type_id"], "label": c["label"],
                "category": c["category"], "category_name": c["category_name"], "icon": c["icon"],
                "detection_confidence": c["detection_confidence"], "commercial_relevance": c["commercial_relevance"],
                "commercial_relevance_level": relevance_level(c["commercial_relevance"]),
                "status": status, "status_label": STATUS_LABELS[status], "identification": ident,
                "seen_count": c["seen_count"], "first_seen": c["first_seen"], "last_seen": c["last_seen"],
                "best": c["best"], "displayed": c["displayed"],
                "occurrences": [{"shot_id": o["shot_id"], "start": o["start"], "end": o["end"]} for o in c.get("occurrences", [])]})
        scenes.append({"scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
                       "context": sc["context"], "objects": objects})
    return {"scenes": scenes, "status_counts": counts, "status_labels": STATUS_LABELS}


def export_view(result: dict) -> dict:
    """Compact, stable JSON for later phases. Every occurrence of an object inherits its identification."""
    out = []
    for sc in result["scenes"]:
        v, e = sc["context"].get("venue"), sc["context"].get("environment")
        objects = []
        for o in sc["objects"]:
            i = o["identification"] or {}
            b, p, var = i.get("brand"), i.get("product"), i.get("variant")
            objects.append({
                "object_id": o["key"], "label": o["label"], "type_id": o["type_id"], "category": o["category"],
                "status": o["status"], "status_label": o["status_label"],
                "brand": b["name"] if b else None, "brand_id": b["id"] if b else None,
                "product": p["name"] if p else None, "product_id": p["id"] if p else None,
                "sku": (var or {}).get("sku") or (p or {}).get("sku"), "url": (var or {}).get("url") or (p or {}).get("url"),
                "variant": (var["name"] or var["sku"]) if var else None, "variant_id": var["id"] if var else None,
                "source": i.get("source"), "identified_by": i.get("identified_by"), "identified_at": i.get("identified_at"),
                "notes": i.get("notes"),
                "detection_confidence": o["detection_confidence"], "commercial_relevance": o["commercial_relevance"],
                "first_seen": o["first_seen"], "last_seen": o["last_seen"], "seen_count": o["seen_count"],
                "occurrences": o["occurrences"]})
        out.append({"scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
                    "context": {"venue": v["label"] if v else None, "environment": e["label"] if e else None},
                    "objects": objects})
    return {"scenes": out}
