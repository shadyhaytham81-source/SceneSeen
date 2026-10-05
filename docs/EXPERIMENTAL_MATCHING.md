# EXPERIMENTAL — DISABLED: Automatic Product Matching

> **This is not part of SceneSeen.** Automatic exact-product matching was built, measured, and then taken out of
> the workflow (2026-10-05) because it is not reliable enough: the right product was ranked first in about two
> thirds of the cases, only about a quarter of real matches reached a trustworthy confidence, and no real product
> catalogue was ever tested. SceneSeen's detector finds generic objects; **people identify products**
> ([IDENTIFICATION.md](IDENTIFICATION.md)).
>
> The code is kept, isolated, in `sceneseen/experimental/matching/` for later research. It:
> - is **not imported** by the server, the CLI or the analysis pipeline (enforced by `tests/test_identification.py`);
> - never runs automatically, never assigns a product, never appears in the UI or in the scene output;
> - adds no latency and loads no model in normal use; its endpoints (`/products/match`, `/products/verify`,
>   `/api/catalog/embed`, `/api/catalog/status`) were removed;
> - can only be run by hand: `scripts/product_matching_study.py` and `scripts/matching_benchmark.py`.
>
> **When to look at it again:** once a few hundred human identifications exist. They are real ground truth
> (object crop → catalogue product), which this study never had. Re-run the study against them; bring matching back
> only as a *suggestion that a person confirms*, and only if "right product in the top 3" and the precision of the
> confident band hold up on real catalogue photos.
>
> Everything below is the original research write-up, unchanged except for file paths. Where it says "the UI",
> "Match Products", "human verification" or "API", it describes the removed integration, not the current product.

```
commercial object (Phase 2A)                               catalogue (docs/CATALOG.md)
  up to 5 occurrence crops ─► embedding + colour signature      reference images ─► embedding + colour signature
                         │                                                       │
                         └──► category gate ─► exact vector search ─► ranked candidates
                                                                          │
                              match confidence (score + lead over the next product)
                                                                          │
                 HIGH CONFIDENCE · POSSIBLE MATCH · UNCERTAIN · NO RELIABLE MATCH
                                                                          │
                              a person confirms / changes / rejects  ─►  audit trail
                                                                          │
                                        products per scene (final output)
```

Nothing is trained. Phase 1 and Phase 2A are not modified: matching reads Phase 2A's cached frames and object
boxes and writes only its own cache and the catalogue database.

**The nearest catalogue item is not "the product".** Unless the evidence is strong, the answer is
"Unknown product", and no product is ever attached to a scene without a person confirming it.

## How the numbers were obtained

There is no labelled product dataset for this project, and no real catalogue yet. The study
(`scripts/product_matching_study.py`) therefore uses the project's own footage:

- 876 object crops from 11 analysed videos;
- 288 pairs of same-type crops from different shots, shown blind (no scores) and labelled *same physical item /
  different / unsure*: 91 same, 94 different, 103 unsure. **The labels are by the AI assistant, not a human**
  (`ground_truth/product_matching/`, same caveat as the Phase 2A reviews);
- 62 objects with at least two confirmed views. One view plays the "catalogue image", the others the detection;
  the distractors are same-type objects from *other* videos (88 on average). Every object is also tested with its
  true product removed, to measure what happens when the product is simply not in the catalogue.

Limits of this protocol, stated plainly: references are film crops, not studio product photos; "same item" inside
one video is easier than a real catalogue with near-identical products; 62 objects is a small sample. The numbers
below are the best available estimate, not a guarantee.

## Embedding model

| model | licence | dim | pair AUC | right product ranked first (gated) | + colour, held-out | extra download |
|---|---|---|---|---|---|---|
| **OpenCLIP ViT-B/32** (laion2b) | MIT | 512 | 0.831 | 44.5 % | 60.4 % | none (Phase 1 already uses it) |
| DINOv2 small | Apache-2.0 | 384 | 0.838 | 45.6 % | 59.3 % | 88 MB |
| DINOv2 base | Apache-2.0 | 768 | 0.830 | 46.2 % | 58.8 % | 346 MB |
| Marqo e-commerce B | Apache-2.0 | 768 | **0.896** | 48.9 % | 61.5 % | 812 MB |
| SigLIP base | Apache-2.0 | 768 | not evaluated | | | the download did not finish in time |
| ensembles of two models | | | 0.84–0.89 | 50–57 % | | two models in memory |

(182 trials each; one percentage point is about two trials. Raw data: `reports/product_matching/embedding_study.json`.)

- **OpenCLIP is sufficient to ship and is the default.** DINOv2 is not better. Its weights are already on disk
  and in memory, so matching adds no download and cannot fail on a model fetch that Phase 1 survived.
- **Marqo e-commerce B is measurably better at ranking** and is the upgrade candidate: with the full pipeline
  (below) it puts the right product first in 73.8 % of cases instead of 67.1 %, with the same precision in the
  high-confidence band. It was trained on product photos, so its advantage on a real catalogue is probably
  larger than this film-crop study shows; that has to be measured with real product images. It is available now
  as a profile: `config/matching_marqo.toml` (own calibrated confidence model).
- **Ensembles** gain 6–8 points for twice the cost and memory. Not adopted.
- Not evaluated: larger CLIP variants (ViT-L, 1.7 GB) and paid embedding APIs (local-first, no paid services).

Changing the embedder never mixes vectors: every cache is keyed by the model.

## Which signals help (measured, OpenCLIP)

| signal | effect on "right product ranked first" | adopted |
|---|---|---|
| Category gating before search | 37.4 % → 44.5 % (and 10× less to compare) | yes |
| Colour (HSV histogram intersection, weight 0.5) | 46.3 % → 66.4 %, the strongest single signal | yes |
| Several occurrences of the object (quality-weighted mean, up to 5) | +1 to +9 points depending on the test | yes |
| Several reference images (best one counts) | by construction; `matched_image` reports which | yes |
| Crop size as a weight (small crops count less) | crops under 160 px: 34 % right, larger: 59 % | yes |
| Lead over the runner-up, for **confidence** | AUC 0.874 vs 0.723 for the raw score | yes (see below) |
| Second embedding model (ensemble) | +6–8 points, double cost | no |
| Scene / shot consistency as its own term | already covered by averaging occurrences | no |
| Text on the product, logos | not attempted | no |

```
score(crop, reference image) = cosine(embeddings) + 0.5 × colour intersection
score(crop, product)         = best of the product's reference images
score(object, product)       = mean over the object's occurrence crops, weighted by crop size
```

## Category gating

A detected *watch* is compared only with products whose object type is `watch`. Types the detector confuses are
gated together (sunglasses / eyeglasses; sofa / armchair / chair; sneakers / shoes; …, the Phase 2A groups). A
product saved without an object type is eligible for every type of its category. If the catalogue has no product
of that kind, the object is a **generic object** and nothing is searched.

## Confidence

A high similarity score alone does not mean a match: lookalikes score high too. What separates a real match is
that it **stands out from the next best product**. The match confidence is a logistic model of both, fitted on
the study and checked leave-one-video-out:

```
match confidence = sigmoid(2.703 × score + 18.441 × (score − best other product) − 4.991)
```

| state | confidence | held-out: top candidate correct | share of cases |
|---|---|---|---|
| **High confidence** | ≥ 0.80 | **93.2 %** (41 / 44) | 15 % |
| Possible match | 0.50 – 0.80 | 58.1 % (18 / 31) | 10 % |
| Uncertain | 0.25 – 0.50 | 50.0 % (25 / 50) | 17 % |
| No reliable match | < 0.25 | 9.2 % (16 / 173) | 58 % |

- Product **not in the catalogue** (149 cases): shown as high confidence in 3 (2.0 %), as uncertain or no match
  in 139 (93.3 %).
- Product **in the catalogue** (149 cases): ranked first in 100 (67.1 %); ranked first *and* high confidence in
  41 (27.5 %).

So the system is deliberately conservative: it is right 9 times out of 10 when it says "high confidence", and it
says so for only about a quarter of the products that are really there. The rest go to a person with ranked
suggestions. Precision was preferred over recall, as required.

**Small catalogues.** With fewer than 5 comparable products there is no meaningful runner-up, so the lead is
measured against the score a wrong product typically reaches (1.047, the 75th percentile in the study) and the
state is capped at "possible match". A catalogue with one watch can never produce a high-confidence watch.

All values are in `[matching]` in `config/default.toml` with the calibration version; a decision stores the
version it was made under. Re-run: `python scripts/product_matching_study.py calibrate`.

## Four separate signals

Every object keeps these apart in the API, the export and the UI:

| field | meaning | source |
|---|---|---|
| `detection_confidence` | the object is there and correctly typed | Phase 2A detector |
| `commercial_relevance` | how commercially interesting it is in this scene | Phase 2A rules |
| `match_confidence` | the best catalogue candidate is this product | this document |
| `verification_status` | `unverified` / `confirmed` / `no_match` | a person |

`status` is only a summary for display: Confirmed · High-confidence candidate · Needs review · Unknown product ·
No match (checked) · Generic object · Not matched yet.

## Human verification

On an object: **Confirm** a candidate, **No match**, **Search catalog** (text search, the object's category by
default), or **Change** to a different product. Each action is stored with who, when, the product, the previous
decision, the candidate's rank and AI confidence at that moment, and the detector / embedder / taxonomy /
calibration versions. `verifications` holds the current decision; `verification_events` is append-only
(History link on each object, `GET /api/videos/{id}/products/history`).

The AI never confirms anything. A decision survives re-analysis because the object key is derived from the
object type, the frame content and the box.

## Final output per scene

`GET /api/videos/{id}/products/export`, the "Export products" button, or `python -m sceneseen products VIDEO`:

```json
{"scene_id": 3, "start_seconds": 61.2, "end_seconds": 118.4,
 "context": {"venue": "Home interior", "environment": "Indoor"},
 "objects": [
  {"label": "Sofa", "type_id": "sofa", "status": "confirmed", "product": "…", "brand": "…", "product_id": 12,
   "detection_confidence": 0.71, "commercial_relevance": 0.52, "match_confidence": 0.94,
   "verification_status": "confirmed", "verified_by": "shady", "verified_at": "2026-10-05T11:04:07",
   "first_seen": 61.2, "last_seen": 110.0, "seen_count": 14},
  {"label": "Watch", "status": "unknown", "product": null, "…": "…"},
  {"label": "Lamp", "status": "no_catalog", "product": null, "…": "…"}]}
```

## Performance and caching

| what | cached by | recomputed when |
|---|---|---|
| catalogue image embedding | image sha256 + model key (database) | never for an unchanged image; a new model version gets its own rows |
| catalogue colour signature | image sha256 + signature version | never |
| object crop embedding | frame content hash + box + model key (`crop_embeddings_<model>.npz` per video) | the frame or box changes |
| search index (in memory) | catalogue fingerprint | the catalogue changes (rebuilt in under a second) |
| ranking | not cached | always recomputed from cached vectors, so it always reflects the current catalogue |

Measured on an Apple M4 (MPS), demo catalogue of 91 products / 190 images, 9 videos, 263 objects:

| step | time |
|---|---|
| Load OpenCLIP (already downloaded) | 6 – 8.5 s once per server process |
| Embed 190 catalogue images | ~5 s once (then never again for an unchanged image) |
| First match of a video (embed 70–160 crops, batch 32, + rank) | 0.2 – 2.1 s |
| Later requests for that video (no model) | 0.04 – 0.36 s, most of it re-assembling Phase 2A's result |
| Embedding throughput | 6 ms per crop (OpenCLIP, batch 32), 28 ms (Marqo) |

Catalogue scaling (`python scripts/matching_benchmark.py`, synthetic 512-d vectors, 3 images per product):

| products | images | index build | index memory | rank one object | rank a 40-object video | without gating |
|---|---|---|---|---|---|---|
| 100 | 300 | 0.01 s | 1 MB | 0.07 ms | 3 ms | 0.17 ms |
| 1,000 | 3,000 | 0.08 s | 9 MB | 0.14 ms | 6 ms | 0.75 ms |
| 10,000 | 30,000 | 0.83 s | 83 MB | 0.63 ms | 38 ms | 5.8 ms |

Search is one matrix product per object (no Python loop over images) and exact. An approximate index or
pgvector becomes worth it somewhere past 10⁵–10⁶ images, or when several servers must share one index.

**Background jobs.** Loading the model, embedding catalogue images, embedding crops and matching run in the
existing single-worker job thread with stage labels (Loading the image model → Reading product images → Reading
the detected objects → Matching products). The page stays responsive and polls `/api/jobs/{id}`. One worker is
deliberate: the detector and the embedder share one GPU. A real queue (RQ / Celery) is the upgrade when there are
several users.

## Failure behaviour

| failure | result |
|---|---|
| catalogue database cannot be opened | catalogue routes answer 503; scenes, Unique Shots, commercial objects work; objects show "Not matched yet" |
| embedding model cannot be loaded / downloaded | job ends with the reason and a "Try again" button; cached matches still shown; commercial objects untouched |
| accelerator error while embedding | retried on CPU |
| a catalogue image file is missing or corrupt | skipped and reported; other images are embedded |
| matching raises for one object | that object is "Not matched yet", the others are ranked |
| no product of that kind / empty catalogue | "Generic object" |
| candidates exist but none is reliable | "Unknown product" |
| corrupt crop cache | ignored and rebuilt |

## Demo run (workflow check, not an accuracy figure)

`python scripts/build_demo_catalog.py --config data/demo/demo.toml` builds 90 demo products from crops of the
analysed videos (brands marked "(demo)", stored only under `data/`, never committed). Matching the same 9 videos:
263 objects → 61 high-confidence candidates, 38 to review, 155 unknown, 9 generic. Of the 61, 58 were the object's
own demo product, 2 the same item in another scene, **1 a product from a different video (wrong)**. All 78 objects
that had their own demo product ranked it first. Because the references are cut from the same footage this is an
easy case; it shows the pipeline works end to end, and the held-out study above is the accuracy estimate.

## Known limits

- **No real catalogue has been tested.** Studio photos on white backgrounds look different from a product worn or
  used in a dark scene. Colour matching masks plain backdrops in catalogue images, but that is untested on real data.
- Colour is the strongest signal, so coloured lighting, black-and-white footage, or two products that differ only
  in a logo or a detail will mislead it.
- Small objects (watch, jewelry, phone) under ~160 px rarely match reliably.
- Clothing is matched as a crop that includes the person and background; there is no segmentation.
- Variants (colour / size) are stored but not matched individually.
- The object's occurrences come from Phase 2A's camera set-ups: if the detection is wrong, the match is wrong.
- Labels behind the calibration are AI-made and the sample is small (62 objects); the thresholds should be
  re-fitted once real confirmations exist (every confirmation stores the score it was made with, for exactly that).
