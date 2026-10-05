# Phase 2B — Human Product Identification

```
Video → Scenes → Unique Shots → Commercial Object Detection → Human Product Identification → Confirmed Scene Products
        (Phase 1)                (Phase 2A, AI: generic objects)  (a person, with the catalogue)
```

**SceneSeen does not guess brands, products, SKUs or variants.** The detector says "sneakers". A person says
"Nike · Air Force 1". Automatic exact-product matching was built and measured, found too unreliable, and removed
from the workflow; it survives only as disabled research code ([EXPERIMENTAL_MATCHING.md](EXPERIMENTAL_MATCHING.md)).

## What the AI still does (unchanged Phase 2A)

Detects generic commercial objects (38 types: smartphone, watch, sneakers, shirt, jacket, handbag, car, bottle,
sofa, lamp, …) with OWLv2 on one frame per unique shot, with bounding boxes, crops, commercial relevance,
occurrence grouping and caching. See [COMMERCIAL.md](COMMERCIAL.md).

## Identifying a product

Every detected object has **Identify Product**. The panel shows:

- **Detected object**, its **scene**, and up to eight **occurrence images**;
- the two AI numbers that describe the detection: detection confidence and commercial relevance (High / Medium / Low);
- the current identification and its **history**;
- **Brand** — search or select; a brand that does not exist can be created on the spot;
- **Product** — search by name or SKU (product or variant SKU), filtered by the chosen brand;
- **Variant** — optional, when the product has variants;
- **Notes**;
- **Confirm Product**, or **Save, confirm later** (recorded, awaiting confirmation), or **Save Brand Only**.

**Product not in catalog** offers two ways out:

- **Create product and link it**: brand, product name, variant, SKU, URL, images, category. The new product goes
  into the catalogue and is linked to the object in the same step.
- **Mark as** Unknown product · Generic / no specific product · Not commercially useful.

### Filtering is not identification

Once the generic object is known, the product list only shows products that object could be. For *sneakers* that
means products typed sneakers or shoes (types the detector confuses are grouped) and fashion products saved
without a type; never a TV, a phone or a sofa. Brands with such products are listed first, with the count.
"Show every category" turns the filter off (useful when the detector's label is wrong). The filter narrows what a
person looks through; it never selects, ranks by likelihood, or suggests anything.

Search is a database query (every typed word must match the name, SKU, variant SKU, external ID or brand), so it
stays fast with thousands of products: nobody scrolls through 5,000 items.

## Statuses

| status | meaning | brand | product |
|---|---|---|---|
| `unidentified` | nobody has looked yet (no stored row) | – | – |
| `brand_identified` | the brand is known, the exact product is not | ✓ | – |
| `product_identified` | a product is selected, not confirmed yet | ✓ | ✓ |
| `confirmed` | a person confirmed the exact product (optionally a variant) | ✓ | ✓ |
| `unknown_product` | a person looked and could not tell which product it is | – | – |
| `no_product` | generic: nothing specific to link | – | – |
| `not_commercial` | not commercially useful | – | – |

`unknown_product` is one more than the six requested: it separates "a person checked and could not tell" from
"nobody has looked". An exact product is never required. The brand always follows the product (a Nike product
cannot be saved under Adidas).

## Editing and history

An identification can be changed or removed at any time (Edit → pick something else, or Remove identification).
Nothing is overwritten silently: `identifications` holds the current statement, and every identify / change /
clear is appended to `identification_events` with the value **before** and **after**, who, when and the note.
History entries store names as they were, so they stay readable after a product is renamed or archived.

## Repeated objects

Phase 2A groups every occurrence of an object type inside a scene into one object ("Sneakers · seen ×6"). The
identification belongs to that grouped object, so it is made once and every occurrence inherits it (the export
lists them). It is **not** copied to other objects that merely share the label: sneakers in another scene, or in
another video, stay unidentified until a person identifies them.

## Final output

`GET /api/videos/{id}/products/export`, the **Export products** button, or `python -m sceneseen products VIDEO`:

```json
{"scene_id": 8, "start_seconds": 412.0, "end_seconds": 468.5,
 "context": {"venue": "Home interior", "environment": "Indoor"},
 "objects": [
  {"object_id": "9f2c…", "label": "Sneakers", "status": "confirmed", "brand": "Nike", "product": "Air Force 1",
   "variant": "White / 42", "sku": "AF1-W-42", "source": "human", "identified_by": "shady",
   "identified_at": "2026-10-05T11:29:57+00:00", "detection_confidence": 0.87, "commercial_relevance": 0.81,
   "seen_count": 6, "occurrences": [{"shot_id": 41, "start": 412.0, "end": 415.2}, "…"]},
  {"label": "Smartphone", "status": "brand_identified", "brand": "Apple", "product": null, "source": "human", "…": "…"},
  {"label": "Watch", "status": "unidentified", "brand": null, "product": null, "source": null, "…": "…"},
  {"label": "Sofa", "status": "not_commercial", "source": "human", "…": "…"}]}
```

AI confidence appears only where an AI decided something: `detection_confidence` and `commercial_relevance`.
There is no product confidence, because no product is chosen by a model.

## Data model and API

Tables `identifications` and `identification_events` ([CATALOG.md](CATALOG.md)); detections are not stored in the
database and are never modified by an identification.

| endpoint | purpose |
|---|---|
| `GET /api/videos/{id}/products` | scenes → objects with status and identification; no model is loaded |
| `POST /api/videos/{id}/products/identify` | `{key, actor, brand_id?, product_id?, variant_id?, status?, confirm?, notes?, clear?}` |
| `GET /api/videos/{id}/products/history?key=` | audit history, newest first |
| `GET /api/videos/{id}/products/export` | the final output above |
| `GET /api/catalog/products?compatible_with=<object type>&q=&brand_id=` | the filtered product search of the panel |
| `GET /api/catalog/brands?compatible_with=<object type>&q=` | the brand search of the panel |

`identify` rules: `actor` is required; `product_id` → confirmed (or product identified with `confirm: false`);
`brand_id` alone → brand identified; `status` ∈ unknown_product / no_product / not_commercial carries no brand or
product; `clear: true` removes the identification. 400 for a rule violation, 404 for an unknown object, product,
brand or variant, 503 when the catalogue database is unavailable.

## Failure behaviour

If the catalogue database cannot be opened, scenes, Unique Shots and commercial objects still work and the objects
are listed as unidentified; saving an identification answers 503. Nothing here can fail because of a model: none
is involved.

## Performance

Identification adds no processing to a video. Reading a video's products is one indexed query plus Phase 2A's
cached result (0.04–0.36 s on the test videos, the same as reading the commercial objects). The server no longer
loads an embedding model for products, embeds catalogue images, or builds a search index.

## Known limits

- **One object per type per scene.** Two different pairs of sneakers in the same scene are one detected object
  (Phase 2A's grouping), so they share one identification. Splitting a grouped object is not built.
- The object key comes from the object type, the frame content and the box of its best detection. If scenes are
  re-cut or the detector settings change, an object can get a new key and show as unidentified; the old
  identification stays in the database and its history, but is not re-attached automatically.
- No accounts: "who" is the name typed in the browser. Fine for one team on one machine, not for audit-grade use.
- No bulk actions ("this brand for all sneakers in this episode") and no keyboard-only flow yet.
- The "product identified → confirmed" step has no separate reviewer role; any user can confirm.
