"""AnalyticsService (FR18-FR19): record views and clicks, report them per brand.

Reports are aggregated: no user ids leave this module (NFR12).
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

from ..schemas import BrandReport, Click, ProductStats, now
from ..store import Store


class AnalyticsService:
    def __init__(self, store: Store):
        self.store = store

    def record_view(self, product_id: str, scan_id: str) -> None:
        self.store.views.append((product_id, scan_id))

    def record_click(self, click: Click) -> None:
        self.store.clicks.append(click)

    def report(self, brand_id: str, period_days: int = 30) -> BrandReport:
        since = now() - timedelta(days=period_days)
        ids = {pid for pid, p in self.store.products.items() if p.brand_id == brand_id}
        recent_scans = {sid for sid, s in self.store.scans.items() if s.created_at >= since}
        views = Counter(pid for pid, sid in self.store.views if pid in ids and sid in recent_scans)
        clicks = Counter(c.product_id for c in self.store.clicks if c.product_id in ids and c.clicked_at >= since)
        rows = [ProductStats(product_id=pid, title=self.store.products[pid].title, views=views[pid], clicks=clicks[pid])
                for pid in ids if views[pid] or clicks[pid]]
        rows.sort(key=lambda r: (r.clicks, r.views), reverse=True)
        return BrandReport(brand_id=brand_id, period_days=period_days, views=sum(views.values()),
                           clicks=sum(clicks.values()), products=rows)
