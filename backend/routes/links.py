"""GET /r/{match_id}: record the click, redirect to the brand's page with UTM tags (FR11)."""
from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from ..auth import current_user
from ..deps import get_analytics, get_store
from ..schemas import Click, User
from ..services.analytics import AnalyticsService
from ..store import Store

router = APIRouter(tags=["links"])


def with_utm(url: str, campaign: str) -> str:
    parts = urlsplit(url)
    utm = urlencode({"utm_source": "sceneseen", "utm_medium": "app", "utm_campaign": campaign})
    return urlunsplit(parts._replace(query=f"{parts.query}&{utm}" if parts.query else utm))


@router.get("/r/{match_id}")
def redirect(match_id: str, user: User | None = Depends(current_user), store: Store = Depends(get_store),
             analytics: AnalyticsService = Depends(get_analytics)):
    m = store.matches.get(match_id)
    if not m:
        raise HTTPException(404, "unknown link")
    analytics.record_click(Click(match_id=m.id, product_id=m.product.id, user_id=user.id if user else None))
    return RedirectResponse(with_utm(m.product.product_url, m.label.value.lower()), status_code=307)
