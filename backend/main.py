"""SceneSeen app API (milestone 2 stubs). Endpoints follow SRS section 3.5.

    uvicorn backend.main:app --reload      ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from .routes import brands, links, recommendations, scans, verified_items

app = FastAPI(title="SceneSeen app API", version="0.2.0",
              description="Milestone 2 stubs: in-memory store, stub detector/embedder.")

for r in (scans.router, links.router, recommendations.router, brands.router, verified_items.router):
    app.include_router(r)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "stage": "milestone-2-stubs"}
