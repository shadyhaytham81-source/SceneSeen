# SceneSeen app backend (milestone 2 stubs)

The API the Expo app will call. This is separate from `server/`, which is the internal video studio. The endpoints follow SRS section 3.5, and the models follow the class diagram in section 3.9.

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload        # interactive docs at http://127.0.0.1:8000/docs
python -m pytest tests/test_backend_stubs.py
```

| Endpoint | FR | Who |
|---|---|---|
| `POST /scans` (image, kind) | FR3–FR8 | anyone |
| `PATCH /scans/{id}` (episode_id) | FR4 | scan owner |
| `GET /scans/{id}/results` | FR6–FR8 | scan owner |
| `GET /r/{match_id}` | FR11 | anyone |
| `GET /recommendations`, `POST/GET /saved` | FR13–FR14 | logged in |
| `POST /brands/{id}/catalogue` (CSV, format) | FR15 | brand partner, admin |
| `POST /verified-items`, `POST /verified-items/{id}/approve` | FR17 | stylist, brand, admin / admin |
| `GET /brands/{id}/report?period=` | FR19 | brand partner, admin |

## What is real and what is a stub

| Part | Status | Milestone 3 |
|---|---|---|
| Routes, roles, request checks | real | — |
| Exact/Similar labels, cutoff, "no close match yet" (`services/matcher.py`) | real | pgvector query |
| Episode suggestion (`services/scene_recogniser.py`) | real, in memory | pgvector query |
| CSV parsing for Shopify, WooCommerce, template (`services/catalogue_importer.py`) | real | download and crop images |
| Analytics and brand report | real, aggregated | — |
| `Detector` | **stub** (finds nothing) | OWLv2 via `sceneseen.commercial.detector` |
| `Embedder` | **stub** (hash vector) | OpenCLIP via `sceneseen.commercial.matching.ClipEncoder` |
| `Recommender` | **stub** (popularity baseline) | model trained on the H&M dataset |
| Auth (`X-User-Id` / `X-User-Role` headers) | **stub** | Supabase Auth JWT |
| Store (`store.py`, in memory) | **stub** | Supabase; schema in `db/schema.sql` |
