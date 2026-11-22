"""Shared service instances, overridable in tests with app.dependency_overrides."""
from __future__ import annotations

from functools import lru_cache

from .services.analytics import AnalyticsService
from .services.embedder import Embedder
from .services.recommender import Recommender
from .services.scan_service import ScanService
from .store import Store, store


def get_store() -> Store:
    return store


@lru_cache
def get_scan_service() -> ScanService:
    return ScanService(store)


def get_embedder() -> Embedder:
    return get_scan_service().embedder


def get_analytics() -> AnalyticsService:
    return AnalyticsService(store)


def get_recommender() -> Recommender:
    return Recommender(store)
