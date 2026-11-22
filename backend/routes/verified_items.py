"""POST /verified-items, POST /verified-items/{id}/approve (FR17)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_role
from ..deps import get_store
from ..schemas import Role, User, VerifiedItem, VerifiedItemCreate, VerifiedStatus
from ..store import Store

router = APIRouter(prefix="/verified-items", tags=["verified items"])


@router.post("", response_model=VerifiedItem, status_code=201)
def confirm(body: VerifiedItemCreate, user: User = Depends(require_role(Role.stylist, Role.brand_partner, Role.admin)),
            store: Store = Depends(get_store)):
    product = store.products.get(body.product_id)
    if not product or body.episode_id not in store.episodes:
        raise HTTPException(404, "unknown product or episode")
    if user.role == Role.brand_partner and product.brand_id != user.brand_id:
        raise HTTPException(403, "brands confirm their own products only")
    item = VerifiedItem(**body.model_dump(), confirmed_by=user.id)
    store.verified[item.id] = item
    return item


@router.post("/{item_id}/approve", response_model=VerifiedItem)
def approve(item_id: str, user: User = Depends(require_role(Role.admin)), store: Store = Depends(get_store)):
    item = store.verified.get(item_id)
    if not item:
        raise HTTPException(404, "unknown verified item")
    item.status, item.approved_by = VerifiedStatus.approved, user.id
    return item
