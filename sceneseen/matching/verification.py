"""Human verification of product matches, with an append-only audit trail.

`verifications` holds the current decision per detected object (confirmed product, or a human
"no match"); `verification_events` records every action ever taken (who, when, what it replaced,
which model versions and confidences were on screen). Nothing here is decided by the AI.
"""
from __future__ import annotations

from sqlalchemy import select

from ..catalog.db import Database, Product, Verification, VerificationEvent, utcnow

ACTIONS = ("confirm", "no_match", "clear")


class VerificationError(ValueError):
    pass


def _row_dict(v: Verification) -> dict:
    p = v.product
    return {
        "status": v.status, "product_id": v.product_id, "variant_id": v.variant_id, "brand_id": v.brand_id,
        "product": None if p is None else {"id": p.id, "name": p.name, "brand": p.brand.name, "sku": p.sku, "url": p.url,
                                          "archived": p.archived,
                                          "image": (f"/api/catalog/images/{p.images[0].sha256}.{p.images[0].ext}"
                                                    if p.images else None)},
        "source": v.source, "suggestion_rank": v.suggestion_rank, "match_score": v.match_score,
        "match_confidence": v.match_confidence, "match_state": v.match_state,
        "detection_confidence": v.detection_confidence, "commercial_relevance": v.commercial_relevance,
        "models": v.models or {}, "decided_by": v.decided_by, "decided_at": v.decided_at.isoformat(),
        "scene_id": v.scene_id, "type_id": v.type_id, "label": v.label}


def decide(db: Database, video_id: str, candidate: dict, action: str, actor: str, product_id: int | None = None,
           variant_id: int | None = None, source: str | None = None, match: dict | None = None,
           models: dict | None = None, video_name: str | None = None) -> dict | None:
    """Record a human decision for one detected object and append it to the audit trail.

    action: confirm (needs product_id) | no_match | clear (remove the decision).
    `match` is the AI ranking that was on screen, so the original confidence is preserved.
    """
    if action not in ACTIONS:
        raise VerificationError(f"action must be one of {ACTIONS}")
    actor = (actor or "").strip()
    if not actor:
        raise VerificationError("a decision needs to say who made it")
    key = candidate["key"]
    with db.session() as s:
        v = s.scalar(select(Verification).where(Verification.video_id == video_id, Verification.candidate_key == key))
        prev_product, prev_status = (v.product_id, v.status) if v else (None, None)
        payload = {"scene_id": candidate.get("scene_id"), "type_id": candidate.get("type_id"),
                   "label": candidate.get("label"), "source": source}
        if action == "clear":
            if v is None:
                return None
            s.delete(v)
            s.add(VerificationEvent(video_id=video_id, candidate_key=key, action="clear", product_id=None,
                                    previous_product_id=prev_product, previous_status=prev_status, actor=actor,
                                    payload=payload))
            return None
        product = None
        rank_, score, conf, state = None, None, None, (match or {}).get("state")
        if action == "confirm":
            if product_id is None:
                raise VerificationError("confirming needs a product")
            product = s.get(Product, int(product_id))
            if product is None:
                raise VerificationError(f"product {product_id} does not exist")
            for c in (match or {}).get("candidates", []):
                if c["product_id"] == product.id:
                    rank_, score, conf = c["rank"], c["score"], c["match_confidence"]
        if v is None:
            v = Verification(video_id=video_id, candidate_key=key, type_id=candidate.get("type_id", ""), decided_by=actor,
                             status="")
            s.add(v)
        v.video_name = video_name
        v.scene_id, v.type_id, v.label = candidate.get("scene_id"), candidate.get("type_id", ""), candidate.get("label")
        v.status = "confirmed" if action == "confirm" else "no_match"
        v.product_id = product.id if product else None
        v.brand_id = product.brand_id if product else None
        v.variant_id = variant_id if product else None
        v.source = source or ("suggestion" if rank_ else "search") if product else None
        v.suggestion_rank, v.match_score, v.match_confidence, v.match_state = rank_, score, conf, state
        v.detection_confidence = candidate.get("detection_confidence")
        v.commercial_relevance = candidate.get("commercial_relevance")
        v.models = models or {}
        v.snapshot = {"best": candidate.get("best"), "first_seen": candidate.get("first_seen"),
                      "last_seen": candidate.get("last_seen"), "seen_count": candidate.get("seen_count"),
                      "top_candidates": [{"product_id": c["product_id"], "score": c["score"],
                                          "match_confidence": c["match_confidence"]}
                                         for c in (match or {}).get("candidates", [])[:3]]}
        v.decided_by, v.decided_at = actor, utcnow()
        changed = prev_status is not None and (prev_product != v.product_id or prev_status != v.status)
        s.add(VerificationEvent(
            video_id=video_id, candidate_key=key, action="change" if changed and action == "confirm" else action,
            product_id=v.product_id, previous_product_id=prev_product, previous_status=prev_status, actor=actor,
            payload={**payload, "suggestion_rank": rank_, "match_score": score, "match_confidence": conf,
                     "match_state": state, "detection_confidence": v.detection_confidence, "models": models or {}}))
        s.flush()
        s.refresh(v)
        return _row_dict(v)


def for_video(db: Database, video_id: str) -> dict[str, dict]:
    """{candidate key: current decision} for one video."""
    with db.session() as s:
        rows = s.scalars(select(Verification).where(Verification.video_id == video_id)).all()
        return {v.candidate_key: _row_dict(v) for v in rows}


def events(db: Database, video_id: str | None = None, candidate_key: str | None = None, limit: int = 200) -> list[dict]:
    """Audit trail, newest first."""
    with db.session() as s:
        q = select(VerificationEvent).order_by(VerificationEvent.id.desc()).limit(limit)
        if video_id:
            q = q.where(VerificationEvent.video_id == video_id)
        if candidate_key:
            q = q.where(VerificationEvent.candidate_key == candidate_key)
        return [{"id": e.id, "video_id": e.video_id, "candidate_key": e.candidate_key, "action": e.action,
                 "product_id": e.product_id, "previous_product_id": e.previous_product_id,
                 "previous_status": e.previous_status, "actor": e.actor, "at": e.at.isoformat(), "payload": e.payload}
                for e in s.scalars(q)]
