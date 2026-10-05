# Phase 2B — Brand & Product Catalogue

The catalogue is what SceneSeen compares detected objects with. It is a normal relational database plus a folder
of image files, managed from the **Catalog** page of the web app.

```
Brand ─┬─ Product ─┬─ ProductVariant        (colour / size / own SKU, optional)
       │           └─ ProductImage ── sha256 ──► image file  +  ImageEmbedding (one per model version)
       └─ …
Verification (current human decision per detected object)   VerificationEvent (append-only audit trail)
```

## Persistence decision

| option | verdict | why |
|---|---|---|
| JSON files | Rejected | no uniqueness rules, no paging or search, no safe concurrent writes, nothing to migrate to |
| **SQLite through SQLAlchemy 2** | **Selected (local default)** | one file, no server, ships with Python; real constraints, indexes and transactions; the same code runs on PostgreSQL |
| PostgreSQL + pgvector | The production target, not a local requirement | needs a running server; nothing in this phase needs it yet (see scaling below) |
| Supabase | Not used | a hosted dependency for a tool that must run offline on a laptop; it is PostgreSQL, so moving there later is the same migration |
| FAISS / hnswlib / a vector DB | Not used | exact search over 30,000 images takes 0.6 ms after category gating (measured below); an approximate index would add a dependency and lose exactness for no gain |

Everything goes through `sceneseen/catalog/db.py` (models) and `sceneseen/catalog/service.py` (operations). No
other module writes SQL. The database URL is `[catalog] database_url` in the config, or the environment variable
`SCENESEEN_DATABASE_URL` (so credentials never go into a file).

**Moving to PostgreSQL / pgvector later:** set the URL to `postgresql+psycopg://…`, install `psycopg`, and the
tables are created as they are. Embeddings are stored as float32 blobs keyed by `(image sha256, model key)`; with
pgvector that column becomes `vector(512)` and `matching/index.py` (one class, ~120 lines) is replaced by an SQL
`ORDER BY embedding <=> :q` query with the same category filter. Nothing else changes. There is no migration tool
yet (`schema_version` is stored in the `meta` table); add Alembic at the first schema change after real data exists.

## Data model

| table | key fields | rules |
|---|---|---|
| `brands` | name, website, description, metadata, archived | name unique (case-insensitive) |
| `products` | brand, name, **category**, **object type**, description, SKU, external ID, URL, price, currency, availability, metadata, archived | name unique per brand; SKU unique per brand; object type must belong to the category |
| `product_variants` | product, name, colour, size, SKU, external ID, URL, price, availability | |
| `product_images` | product, optional variant, **role** (front / back / side / detail / lifestyle / other), sha256, size, position | the same picture cannot be attached twice to one product |
| `image_embeddings` | (sha256, model key) → vector | one row per image content and model version |
| `verifications` | (video, object key) → confirmed product or "no match", who, when, AI confidence at that moment, model versions | one current decision per object |
| `verification_events` | every confirm / change / no match / clear, with the previous value | append-only |

Categories and object types are the Phase 2A taxonomy (`commercial/taxonomy.py`), so a product's type is the same
vocabulary the detector uses. That is what makes category gating possible.

**Images** are stored content-addressed: `data/catalog/images/<aa>/<sha256>.<ext>`. Uploads are decoded and
validated (JPEG / PNG / WebP, at least 32 px, at most 25 MB); the stored name never comes from the user. The same
picture used by two products is one file and one embedding. Deleting an image detaches it; the file and its
embedding stay, so nothing has to be recomputed if it is attached again.

**Links** (product URL, brand website) must be `http://` or `https://`; anything else is rejected, because they
are shown as clickable links.

**Deleting** a product that a person has already confirmed in a video archives it instead, so the audit trail
never points at nothing. Archived products and brands are hidden from matching and can be restored.

## Catalog page

Search (name, SKU, external ID, brand), category chips with counts, brand filter, archived toggle, paging (24 per
page). "Add product" opens one form: brand (type to pick or create), name, category, object type, SKU, external
ID, URL, price, currency, availability, description, and a drop zone for several images with a role each.
Brands are renamed / archived in the Brands dialog. Search and paging run in the database:

| catalogue size | text search | one page of a category |
|---|---|---|
| 100 products | 1.2 ms | 0.3 ms |
| 1,000 | 2.0 ms | 1.5 ms |
| 10,000 | 10.3 ms | 3.4 ms |

(`python scripts/matching_benchmark.py`, Apple M4, SQLite.)

## API

| endpoint | purpose |
|---|---|
| `GET /api/catalog/meta` | categories, object types, image roles, counts |
| `GET/POST /api/catalog/brands`, `PATCH /api/catalog/brands/{id}` | brands |
| `GET /api/catalog/products?q=&category=&object_type=&brand_id=&archived=&limit=&offset=` | paged search |
| `POST /api/catalog/products`, `GET/PATCH/DELETE /api/catalog/products/{id}` | products |
| `POST /api/catalog/products/{id}/variants`, `PATCH/DELETE /api/catalog/variants/{id}` | variants |
| `POST /api/catalog/products/{id}/images` (multipart, several files + role) | upload; a bad file is reported, good ones are kept |
| `PATCH/DELETE /api/catalog/images/{id}`, `GET /api/catalog/images/{sha256}.{ext}?size=` | image role / removal / file |
| `GET /api/catalog/status`, `POST /api/catalog/embed` | images waiting for an embedding; embed them now (background job) |

Errors: 400 invalid input, 404 not found, 409 duplicate, **503 catalogue unavailable**. A 503 affects only these
routes; scenes, Unique Shots and commercial objects keep working.

## Not built
Bulk import (CSV / Shopify / feed), user accounts and permissions, image background removal, per-variant
matching (variants are stored, matching is per product). There is no authentication: this is a local tool.
