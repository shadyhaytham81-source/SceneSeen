"""Final Phase 2 output: the products of each scene.

Combines, without collapsing them:
    commercial objects (Phase 2A)  +  catalogue candidates (matcher)  +  human decisions (verification)

Each object keeps four separate fields (detection_confidence, commercial_relevance,
match_confidence, verification_status) plus one overall `status` for display:

    confirmed                  a human confirmed a catalogue product
    high_confidence_candidate  the matcher is confident; not confirmed yet
    needs_review               possible / uncertain candidates exist
    unknown                    nothing in the catalogue is a reliable match  -> "Unknown product"
    no_match_confirmed         a human checked and said: none of the catalogue products
    no_catalog                 the catalogue has no products of this kind    -> "Generic object"
    not_matched                matching has not been run (or is unavailable)
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..catalog.db import Database, Product

STATUS_LABELS = {
    "confirmed": "Confirmed", "high_confidence_candidate": "High-confidence candidate", "needs_review": "Needs review",
    "unknown": "Unknown product", "no_match_confirmed": "No match (checked)", "no_catalog": "Generic object",
    "not_matched": "Not matched yet"}


def products_brief(db: Database, ids: set[int]) -> dict[int, dict]:
    if not ids:
        return {}
    with db.session() as s:
        rows = s.scalars(select(Product).where(Product.id.in_(ids)).options(
            selectinload(Product.brand), selectinload(Product.images))).all()
        return {p.id: {"id": p.id, "name": p.name, "brand": p.brand.name, "brand_id": p.brand_id, "sku": p.sku,
                       "url": p.url, "category": p.category, "object_type": p.object_type, "archived": p.archived,
                       "image": f"/api/catalog/images/{p.images[0].sha256}.{p.images[0].ext}" if p.images else None,
                       "images": {i.id: f"/api/catalog/images/{i.sha256}.{i.ext}" for i in p.images}}
                for p in rows}


def overall_status(match: dict | None, verification: dict | None) -> str:
    if verification:
        return "confirmed" if verification["status"] == "confirmed" else "no_match_confirmed"
    if match is None:
        return "not_matched"
    if match.get("eligible_products", 0) == 0:
        return "no_catalog"
    return {"high_confidence": "high_confidence_candidate", "possible_match": "needs_review",
            "uncertain": "needs_review"}.get(match["state"], "unknown")


def scene_products(commercial: dict, match_rec: dict | None, verifications: dict[str, dict], db: Database | None,
                   include_hidden: bool = False) -> dict:
    """Per scene: context + every commercial object with its product status."""
    matches = (match_rec or {}).get("matches", {})
    ids = {c["product_id"] for m in matches.values() for c in m.get("candidates", [])}
    brief = products_brief(db, ids) if db is not None else {}
    scenes, counts = [], {k: 0 for k in STATUS_LABELS}
    for sc in commercial.get("scenes", []):
        objects = []
        for c in sc["candidates"]:
            if not c["displayed"] and not include_hidden:
                continue
            m = matches.get(c["key"])
            v = verifications.get(c["key"])
            status = overall_status(m, v) if c["displayed"] else "not_matched"
            cands = []
            for k in (m or {}).get("candidates", []):
                p = brief.get(k["product_id"])
                if p is None:
                    continue
                cands.append({**k, "product": {x: p[x] for x in ("id", "name", "brand", "sku", "url", "image", "archived")},
                              "matched_image": p["images"].get(k.get("image_id"), p["image"])})
            counts[status] += 1
            objects.append({
                "key": c["key"], "type_id": c["type_id"], "label": c["label"], "category": c["category"],
                "category_name": c["category_name"], "icon": c["icon"],
                "detection_confidence": c["detection_confidence"], "commercial_relevance": c["commercial_relevance"],
                "match_confidence": (m or {}).get("match_confidence"),
                "match_state": (m or {}).get("state"), "match_note": (m or {}).get("reason"),
                "verification_status": v["status"] if v else "unverified",
                "status": status, "status_label": STATUS_LABELS[status],
                "product": v["product"] if v and v["status"] == "confirmed" else None,
                "verification": v, "candidates": cands,
                "eligible_products": (m or {}).get("eligible_products"),
                "seen_count": c["seen_count"], "first_seen": c["first_seen"], "last_seen": c["last_seen"],
                "best": c["best"], "displayed": c["displayed"], "review": c.get("review")})
        scenes.append({"scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
                       "context": sc["context"], "objects": objects})
    return {"scenes": scenes, "status_counts": counts, "status_labels": STATUS_LABELS}


def export_view(result: dict) -> dict:
    """Compact, stable JSON for later phases: one line of truth per object."""
    out = []
    for sc in result["scenes"]:
        v, e = sc["context"].get("venue"), sc["context"].get("environment")
        out.append({
            "scene_id": sc["scene_id"], "start_seconds": sc["start_seconds"], "end_seconds": sc["end_seconds"],
            "context": {"venue": v["label"] if v else None, "environment": e["label"] if e else None},
            "objects": [{
                "label": o["label"], "type_id": o["type_id"], "category": o["category"], "status": o["status"],
                "product_id": o["product"]["id"] if o["product"] else None,
                "product": o["product"]["name"] if o["product"] else None,
                "brand": o["product"]["brand"] if o["product"] else None,
                "detection_confidence": o["detection_confidence"], "commercial_relevance": o["commercial_relevance"],
                "match_confidence": o["match_confidence"], "verification_status": o["verification_status"],
                "verified_by": o["verification"]["decided_by"] if o["verification"] else None,
                "verified_at": o["verification"]["decided_at"] if o["verification"] else None,
                "first_seen": o["first_seen"], "last_seen": o["last_seen"], "seen_count": o["seen_count"]}
                for o in sc["objects"] if o["displayed"]]})
    return {"scenes": out}
