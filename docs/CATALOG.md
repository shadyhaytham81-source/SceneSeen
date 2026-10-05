# Phase 2B — Brand & Product Catalogue

The catalogue is the source of truth for brands and products. People link detected objects to it
([IDENTIFICATION.md](IDENTIFICATION.md)); SceneSeen never picks a product by itself. It is a normal relational
database plus a folder of image files, managed from the **Catalog** page of the web app.

```
Brand ─┬─ Product ─┬─ ProductVariant        (colour / size / own SKU, optional)
       │           └─ ProductImage ── sha256 ──► image file
       └─ …
Identification (what a person says a detected object is)    IdentificationEvent (append-only history)
```

## Persistence decision

| option | verdict | why |
|---|---|---|
| JSON files | Rejected | no uniqueness rules, no paging or search, no safe concurrent writes, nothing to migrate to |
| **SQLite through SQLAlchemy 2** | **Selected (local default)** | one file, no server, ships with Python; real constraints, indexes and transactions; the same code runs on PostgreSQL |
| PostgreSQL | The production target, not a local requirement | needs a running server; nothing in this phase needs it yet |
| Supabase | Not used | a hosted dependency for a tool that must run offline on a laptop; it is PostgreSQL, so moving there later is the same migration |
| A vector database | Not used | nothing in the workflow searches by image; products are found by text search and filters |

Everything goes through `sceneseen/catalog/db.py` (models), `service.py` (catalogue operations) and
`identification.py` (human identifications). No other module writes SQL. The database URL is
`[catalog] database_url` in the config, or the environment variable `SCENESEEN_DATABASE_URL` (so credentials never
go into a file).

**Moving to PostgreSQL later:** set the URL to `postgresql+psycopg://…`, install `psycopg`, and the tables are
created as they are. There is no migration tool yet. `schema_version` is stored in the `meta` table (now 2); the
only change so far was additive (the identification tables), which `create_all` handles on an existing database.
Add Alembic at the first non-additive change after real data exists. The `verifications` / `verification_events`
tables of schema 1 are no longer used or created; if an old local database has them they are left alone.

## Data model

| table | key fields | rules |
|---|---|---|
| `brands` | name, website, description, metadata, archived | name unique (case-insensitive) |
| `products` | brand, name, **category**, **object type**, description, SKU, external ID, URL, price, currency, availability, metadata, archived | name unique per brand; SKU unique per brand; object type must belong to the category |
| `product_variants` | product, name, colour, size, SKU, external ID, URL, price, availability | |
| `product_images` | product, optional variant, **role** (front / back / side / detail / lifestyle / other), sha256, size, position | the same picture cannot be attached twice to one product |
| `identifications` | (video, object key) → status, brand?, product?, variant?, notes, source = human, who, when | one current row per detected object; no row = unidentified |
| `identification_events` | every identify / change / clear, with the value before and after | append-only |
| `image_embeddings` | (sha256, model key) → vector | empty in normal use; only the disabled experimental matcher writes it |

Detections are **not** in this database. They stay in the per-video analysis cache, produced by the model; the
database only holds what people entered.

Categories and object types are the Phase 2A taxonomy (`commercial/taxonomy.py`), so a product's type is the same
vocabulary the detector uses. That is what lets the identification panel list only products a detected object
could be.

**Images** are stored content-addressed: `data/catalog/images/<aa>/<sha256>.<ext>`. Uploads are decoded and
validated (JPEG / PNG / WebP, at least 32 px, at most 25 MB); the stored name never comes from the user. The same
picture used by two products is one file. Deleting an image detaches it; the file stays.

**Links** (product URL, brand website) must be `http://` or `https://`; anything else is rejected, because they
are shown as clickable links.

**Deleting** a product that a person has identified in a video archives it instead, so identifications and their
history never point at nothing; a variant in use cannot be deleted. Archived products and brands are hidden from
search and can be restored.

## Catalog page

Search (name, SKU, variant SKU, external ID, brand; every word must match), category chips with counts, brand
filter, archived toggle, paging (24 per page). "Add product" opens one form: brand (type to pick or create), name,
category, object type, SKU, external ID, URL, price, currency, availability, description, and a drop zone for
several images with a role each. Brands are renamed / archived in the Brands dialog. Search and paging run in the
database:

| catalogue size | text search | one page of a category |
|---|---|---|
| 100 products | 1.2 ms | 0.3 ms |
| 1,000 | 2.0 ms | 1.5 ms |
| 10,000 | 10.3 ms | 3.4 ms |

(Apple M4, SQLite; measured before variant-SKU search was added to the query.)

## API

| endpoint | purpose |
|---|---|
| `GET /api/catalog/meta` | categories, object types, image roles, counts |
| `GET /api/catalog/brands?q=&compatible_with=&limit=`, `POST /api/catalog/brands`, `PATCH /api/catalog/brands/{id}` | brands; with `compatible_with`, counts only products that object type could be and lists those brands first |
| `GET /api/catalog/products?q=&category=&object_type=&compatible_with=&brand_id=&archived=&limit=&offset=` | paged search; `compatible_with=sneakers` keeps footwear and drops TVs |
| `POST /api/catalog/products`, `GET/PATCH/DELETE /api/catalog/products/{id}` | products |
| `POST /api/catalog/products/{id}/variants`, `PATCH/DELETE /api/catalog/variants/{id}` | variants |
| `POST /api/catalog/products/{id}/images` (multipart, several files + role) | upload; a bad file is reported, good ones are kept |
| `PATCH/DELETE /api/catalog/images/{id}`, `GET /api/catalog/images/{sha256}.{ext}?size=` | image role / removal / file |

Errors: 400 invalid input, 404 not found, 409 duplicate, **503 catalogue unavailable**. A 503 affects only the
catalogue and identification routes; scenes, Unique Shots and commercial objects keep working.

## Not built
Bulk import (CSV / Shopify / feed), user accounts and permissions. There is no authentication: this is a local
tool, and the name stored with an identification is whatever the person typed.
