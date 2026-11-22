"""POST /brands/{id}/catalogue (FR15), GET /brands/{id}/report (FR19)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..auth import require_role
from ..deps import get_analytics, get_embedder, get_store
from ..schemas import BrandReport, ImportReport, Role, User
from ..services.analytics import AnalyticsService
from ..services.catalogue_importer import IMPORTERS, ShopifyImporter
from ..services.embedder import Embedder
from ..store import Store

router = APIRouter(prefix="/brands", tags=["brands"])
brand_staff = require_role(Role.brand_partner, Role.admin)


def _check_brand(store: Store, brand_id: str, user: User) -> None:
    if brand_id not in store.brands:
        raise HTTPException(404, "unknown brand")
    if user.role == Role.brand_partner and user.brand_id != brand_id:
        raise HTTPException(403, "not your brand")


@router.post("/{brand_id}/catalogue", response_model=ImportReport, status_code=201)
async def import_catalogue(brand_id: str, file: UploadFile = File(...), format: str = Form("template"),
                           store_url: str = Form(""), user: User = Depends(brand_staff),
                           store: Store = Depends(get_store), embedder: Embedder = Depends(get_embedder)):
    _check_brand(store, brand_id, user)
    cls = IMPORTERS.get(format)
    if not cls:
        raise HTTPException(400, f"format must be one of {sorted(IMPORTERS)}")
    importer = cls(store, embedder, store_url) if cls is ShopifyImporter else cls(store, embedder)
    try:
        return importer.run(await file.read(), brand_id)
    except UnicodeDecodeError:
        raise HTTPException(400, "file must be a UTF-8 CSV")


@router.get("/{brand_id}/report", response_model=BrandReport)
def report(brand_id: str, period: int = 30, user: User = Depends(brand_staff), store: Store = Depends(get_store),
           analytics: AnalyticsService = Depends(get_analytics)):
    _check_brand(store, brand_id, user)
    return analytics.report(brand_id, period_days=min(max(period, 1), 365))
