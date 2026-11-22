"""GET /recommendations, saved items (FR13-FR14)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_login
from ..deps import get_recommender, get_store
from ..schemas import Product, SavedItem, User
from ..services.recommender import Recommender
from ..store import Store

router = APIRouter(tags=["viewer"])


@router.get("/recommendations", response_model=list[Product])
def recommendations(k: int = 12, user: User = Depends(require_login), rec: Recommender = Depends(get_recommender)):
    return rec.recommend(user.id, k=min(max(k, 1), 50))


@router.post("/saved/{product_id}", response_model=SavedItem, status_code=201)
def save_item(product_id: str, user: User = Depends(require_login), store: Store = Depends(get_store)):
    if product_id not in store.products:
        raise HTTPException(404, "unknown product")
    item = SavedItem(user_id=user.id, product_id=product_id)
    store.saved.append(item)
    return item


@router.get("/saved", response_model=list[SavedItem])
def saved(user: User = Depends(require_login), store: Store = Depends(get_store)):
    return [s for s in store.saved if s.user_id == user.id]
