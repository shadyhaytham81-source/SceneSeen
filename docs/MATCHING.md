# Phase 2B — Product Catalogue Matching

Connects SceneSeen's detected commercial objects (Phase 2A) to brand catalogues, starting with
**IN YOUR SHOE (IYS)**: 1,183 products, about 5,460 images.

```
Video → Scenes → Unique Shots → OWLv2 detection ("Black jacket", type jacket, colour black)
                                        │
                     1. normalise the label   (product_taxonomy.json)
                     2. product types it may be: strict = jacket, overshirt, blazer, coat
                                                 broad  = hoodie, sweatshirt, cardigan, …
                     3. keep only catalogue products of those types
                                        │
                     4. OpenCLIP embedding of the detected crop (box + 8 % padding)
                     5. cosine similarity with the GARMENT CROP of every photo of the remaining products
                        → product score = mean of its two closest photos
                        → colour adjustment, scaled by how clear the detected colour is
                                        ▼
                     top 3 products: image, name, brand, price, similarity

Catalogue side, built once:  catalogue photo ──OWLv2, prompted with the product's own type──▶ garment box
                             ──same 8 % padding──▶ crop (saved under crops/) ──OpenCLIP──▶ embeddings_garment.npz
```

## Why the taxonomy filter

Catalogue photos show models wearing whole outfits: the photo of "Black Wide Leg Pants" also shows a white shirt.
Image similarity alone happily matches a detected shirt to those pants. So the catalogue is filtered to
**compatible product types first**, and similarity only ranks what is left.

Measured on 263 real OWLv2 detections from 9 analysed videos against a 130-product IYS sample
(2026-10-27, OpenCLIP ViT-B/32):

| | before (similarity only) | after (filter, then similarity) |
|---|---|---|
| clothing objects: top-3 slots of an incompatible garment type | 322 of 444 (73 %) | 0 |
| non-clothing objects (lamp, sofa, phone, car…): unrelated products shown | 345 of 345 slots | none shown |
| clothing objects whose first candidate changed | — | 69 of 148 |

This measures *type* correctness only. Whether the candidate is the exact product on screen has not been measured
(there is no labelled ground truth of products in these videos), and the full catalogue will give different numbers.

## Garment-level catalogue crops

Filtering by type removes wrong-type *products*, but each remaining product was still represented by the whole
photo: a pair of trousers by a model in a shirt, trousers and shoes. Each catalogue photo is therefore cropped to the
garment being sold before it is embedded (`sceneseen/commercial/garment_crop.py`):

- **Detector:** OWLv2, the model Phase 2A already uses for video, so nothing new is installed. It is prompted with
  the product's own type (`crop` in `product_taxonomy.json`): jeans → "jeans", "pants"; cardigan → "cardigan",
  "sweater"; bucket hat → "bucket hat", "hat"; socks → "socks". Video crops come from OWLv2 too, so both sides are
  "OWLv2 box + 8 % padding".
- **Every prompt is scored on its own.** Standard OWLv2 post-processing gives each box only its best prompt; with
  ~55 garment prompts a cardigan came back as "knit top" and 29 of 84 sample photos found nothing. Ranking boxes
  per prompt brought that to 7 of 84 (candles, towels, one crochet top).
- **Selection:** the highest-scoring box of the product's prompts with score ≥ `crop_min_score` (0.15).
  - Sets and pyjamas: best top box ∪ best bottom box.
  - Clothing boxes under 2 % of the photo are ignored (a collar is not the polo).
  - Layering: an open shirt over a tee is found as "shirt jacket" while its "shirt" box is the tee, so for shirts a
    higher-scoring containing "shirt jacket" box wins.
  - No confident box, or unknown product type → the **full photo is kept** and recorded (`crop_method: full`). Nothing
    is centre-cropped blindly.
- **Measured on IN YOUR SHOE (250 products, 1,222 photos):** 917 single garment boxes, 210 unions (sets),
  95 full photos. Visual check of 48 random crops (AI-judged): 44 correct, 2 partial, 2 wrong. The 2 wrong ones were
  open shirts over tees, which led to the layering rule.
- **Not letterboxed.** Padding crops to a square (so CLIP's centre crop keeps a whole trouser leg) was measured and
  is clearly worse: top-1 79 % → 67 % (see Evaluation). Crops go through CLIP's normal resize + centre crop, the same
  as video crops. `letterbox = true` remains available and is part of the index version.

Check the crops yourself at any time:

```bash
python -m sceneseen match-catalogue --inspect-crops 48     # <brand>/crop_inspection.jpg: red = garment box, blue = full photo
```

## Rules

- **OWLv2 detects; the taxonomy decides what is comparable; embeddings rank.** Neither replaces the other.
- **Strict first, broader only if needed.** If fewer than 3 products of a strict type exist, the remaining slots are
  filled from the *broad* types (marked "broader category" in the UI, logged as a fallback). Example: IYS sells no
  backpacks, so a detected backpack is compared with tote bags, marked as broader.
- **Never unrelated products.** Object types with no compatible product type (car, lamp, phone, sofa) get no
  candidates, with the reason. Products whose type is unknown are never offered.
- **Colour ranks, it does not filter.** The score moves by up to +0.04 (same colour), +0.02 (close), −0.04 (different)
  or −0.01 (different, multicolour product). A good match in another colour is moved down, never removed.
  - The adjustment is scaled by how clearly the detected crop shows its colour (`color_confidence`: share of the box
    centre in that colour, 55 % → 0.29 … ≥ 80 % → 1). An uncertain colour barely moves anything; no colour, none.
  - Product colour comes from the name / colour option first. Products whose name has no colour (most printed
    tees) use the dominant colour of their garment crop at **half** weight (`color_source: image`).
  - Weights are a heuristic, not calibrated: `[catalogue] color_*` in `config/default.toml`.
- **Product score = mean similarity of its two closest photos** (`aggregate = "top2"`; one photo → that photo).
  A product whose several photos agree outranks one lucky photo. `max` is the previous behaviour.
- **Similarity is not a probability.** It is the cosine similarity of the crop and the product's closest image,
  and its scale depends on the representation (garment crops score higher than whole photos). Each candidate also
  carries `relative_score` (its rank score minus the first candidate's) for comparing within one list only. The UI
  shows the number without calling it a percentage or a "match".

## Taxonomy (`sceneseen/commercial/product_taxonomy.json`)

Built from the real IYS product types (77 seen in a 250-product sample, including shop jargon: "Pjoys" = pyjamas,
"Swants" = sweatpants, "Sworts" = sweat shorts, "Jorts" = denim shorts, "Neck Socks"). It has:

- `product_types`: 48 types, each in one umbrella (tops, upper_layers, outerwear, trousers, shorts, skirts_dresses,
  sets, sleepwear, one_piece, headwear, footwear, socks, bags, accessories, home), with keywords;
- `type_order`: specific types first (overshirt before shirt, swim shorts before shorts, polo before t-shirt);
- `detections`: for each OWLv2 type, its `strict` and `broad` product types;
- `detection_label_synonyms`: free-text labels from other detectors ("black pants" → trousers);
- `colors` / `near_colors`: colour words → 12 colour families (the ones the detector reports);
- `ambiguous_keywords`: words deliberately not classified.

Rules: the shop's `product_type` is read first, the product name second; matching is on whole words ("Short
Sleeve Shirt" is a shirt, "Short Socks" are socks). The original product name is always kept. Anything
ambiguous stays unknown: IYS "Flowy Wraps" (wrap skirt or beach wrap?) are not offered until someone decides.
Colours are read only from clear colour words; motif words (cherry, coffee, mint) are not colours because
printed designs use them, so printed tees usually have no colour.

To extend it, edit the JSON (no code change, no re-embedding) and check with:

```bash
python -m sceneseen match-catalogue --taxonomy-report        # products per type, and every UNKNOWN product type
```

## Embeddings

- Model: **OpenCLIP ViT-B/32 (laion2b_s34b_b79k)**, the model Phase 1 already uses. The same model and the same
  OpenCLIP preprocessing embed catalogue crops and detected crops (crop = box + 8 % padding on both sides).
  The model was not replaced: the evaluation below shows the representation, not the model, was the problem to
  fix first.
- `[catalogue] image_mode` picks the representation: `garment` (default) or `full` (whole photos, the original
  index). **Each mode has its own files**, and the mode is recorded and checked on load, so the two are never mixed.
  A brand whose garment index is not built yet uses its full-photo index meanwhile, with a warning, so matching keeps
  working.
- Files per brand (next to `catalogue.jsonl`):

| file | content |
|---|---|
| `embeddings_garment.npz` / `embeddings.npz` | float32 `[images, 512]`, L2-normalised; row *i* belongs to manifest record *i* |
| `embeddings_garment_manifest.json` / `embeddings_manifest.json` | per row: brand, product id, handle, name, product type, price, currency, product URL, original image path, image path for serving, source CDN URL, file size + mtime; garment mode adds `crop_path`, `crop_box`, `crop_method`, `crop_score`, `crop_prompt`, `crop_color` |
| `embeddings_garment_meta.json` / `embeddings_meta.json` | image mode, model, pretrained weights, preprocessing fingerprint, dimension, counts, build time; garment mode adds the crop settings (detector, padding, letterbox, min score) and how many photos got each crop method |
| `garment_boxes.json` | OWLv2 detections of every photo for every crop prompt, keyed by file size + mtime (the slow step, cached) |
| `crops/<brand>/<handle>/<file>.jpg` | the garment crops that were embedded (long side 512 px) |

- **Nothing is regenerated on a normal run.** Building is a separate command and only embeds what changed:

| change | what happens |
|---|---|
| new images or products | only those are embedded |
| an image file replaced (size or modification time changed) | only that image is re-embedded |
| an image deleted | its row is dropped |
| product name / price / type changed in `catalogue.jsonl` | no re-embedding, except a garment photo whose selected box changes because its type changed; the manifest is refreshed on the next build |
| taxonomy edited | no re-embedding (garment mode: only photos whose selected garment box changed); cached match results are recomputed automatically. Changing a crop **prompt** re-runs garment detection on every photo |
| embedding model, weights or preprocessing changed; garment padding / letterbox changed | **everything must be re-embedded**; matching refuses the old embeddings and says so |

- Old indexes made before `embeddings_meta.json` existed (the first version of this matcher) are accepted as
  ViT-B/32 / laion2b, and the next `--build-index` records their version **without re-embedding**.
- Each brand is its own folder with its own index, so adding a brand never rebuilds the others.

### Regenerating the embeddings locally

```bash
python -m sceneseen match-catalogue                         # status: which index each brand uses, what is stale
python -m sceneseen match-catalogue --build-index           # garment index: detect garments in new photos, crop, embed
python -m sceneseen match-catalogue --build-index --mode all            # garment AND full-photo index (for the evaluation)
python -m sceneseen match-catalogue --build-index --brand in_your_shoe --force   # re-embed everything
python -m sceneseen match-catalogue --verify 50             # re-embed 50 sampled rows: do they still match their records/crops?
python -m sceneseen match-catalogue --inspect-crops 48      # contact sheet of sampled garment crops
```

- **Garment detection is the slow part:** OWLv2 on each catalogue photo once, about 1 s per photo on an Apple M4
  (MPS). That is ~1.5 h for the 5,461 IYS photos. Plug the laptop in and run it in a terminal you keep open:
  macOS throttles the GPU for background processes and on battery (measured 1.8–3.7 s per photo).
  Interrupting is safe: detections are saved as they go and the next `--build-index` continues where it stopped.
- **Embedding the crops** is fast, at roughly 50 photos/s (about 2 minutes for IYS).
- **Loading an existing index** takes well under a second.
- The OWLv2 weights (~620 MB) are the ones Phase 2A already downloads; nothing else is installed.

## Where the dataset goes (it is not in Git)

The scraped catalogue, its images and the embeddings are **never committed** (`.gitignore` covers `scraper/output/`,
`data/`, and `catalogue.jsonl`, `embeddings*.npz/json` anywhere). Each person keeps their own copy:

```
<catalogue folder>/
  in_your_shoe/
    catalogue.jsonl
    catalogue.csv                       (optional)
    images/in_your_shoe/<handle>/<file>.jpg     (images/<handle>/<file> also works)
    embeddings.npz, embeddings_manifest.json, embeddings_meta.json   (created by --build-index)
  another_brand/ …
```

SceneSeen looks in `scraper/output/` and `data/catalogues/` inside the project (where the scraper writes), or
anywhere you point it:

```bash
export SCENESEEN_CATALOGUE_DIR="/path/to/folder/that/contains/in_your_shoe"     # macOS / Linux
$env:SCENESEEN_CATALOGUE_DIR = "D:\datasets\catalogues"                         # Windows PowerShell
```

(or set `[catalogue] dirs` in a config override file). Several folders: separate with `:` (`;` on Windows).

If the dataset or the embeddings are missing or unusable, the Product Matching tab and `commercial --match` say
exactly that and what to run; scenes, unique shots and detected objects keep working.

| status | meaning | fix |
|---|---|---|
| `missing_dataset` | no brand folder with `catalogue.jsonl` in the searched folders | copy the dataset there or set `SCENESEEN_CATALOGUE_DIR` |
| `not_built` | catalogue found, no embeddings | `python -m sceneseen match-catalogue --build-index` |
| `incompatible` | embeddings made with another model / preprocessing, or files that no longer belong together | `… --build-index --force` |
| `ready` + warnings | images added / changed / removed since the build | `… --build-index` (matching keeps working meanwhile) |

## Web UI

Run SceneSeen as usual and open a video's **Commercial Objects** section:

- **Product Matching** tab: every detected object next to its top 3 candidates (image, brand, name, price,
  similarity, product type, colour agreement, "broader category" marker), and a line saying which product types it
  was compared with and which colour was detected.
- **Detected Objects** tab: clicking an object also shows its candidates under the frame.
- Matches are computed the first time the section is opened (this loads the image model once), then cached in
  `data/cache/<video>/commercial/matches.json` and recomputed automatically when the catalogue index, the taxonomy
  or the matching settings change.

## API

| endpoint | description |
|---|---|
| `GET /api/videos/{vid}/commercial/matches?compute=true&reload=false` | `{status, message, warnings, brands, products, images, matches: {object key: {query, candidates, fallback, pool, note}}}`; `compute=false` only reads the cache; `reload=true` re-reads the dataset from disk |
| `GET /api/catalogue/status` | which brands were found and whether their embeddings are usable |
| `GET /api/catalogue/image?brand=<folder>&rel=<path>&size=320` | a catalogue image; only files inside the configured catalogue folders are served |

## CLI

```bash
python -m sceneseen commercial path/to/video.mp4 --match    # detected objects + their catalogue candidates
python -m sceneseen match-catalogue --taxonomy-report
```

## Evaluation (`scripts/product_matching_eval.py`)

There is no ground truth of which IYS product appears in the videos, so two checks were run on the real data
(2026-10-29, OpenCLIP ViT-B/32; report: `reports/matching/eval_20261029-233715.json`):

**1. Catalogue retrieval with ground truth (leave one photo out).**
- **Queries:** for each product with ≥ 2 photos, one photo is cut to its OWLv2 garment box + padding (like a video
  detection) and removed from the index. The product's other photos have to bring it back.
- **Ranking:** production code (type filter, colour, top 3).
- **Data:** 229 queries over the 250 IYS products whose photos had been garment-detected at the time (all garment
  kinds, including pyjamas and sets).

| variant | top-1 | top-3 | trousers top-1 (n = 32) | tops top-1 (n = 62) |
|---|---|---|---|---|
| A. before: whole catalogue photos, old ranking | 79.0 % | 93.4 % | 53.1 % | 80.6 % |
| B. garment crops, old ranking | 79.0 % | 93.0 % | 62.5 % | 80.6 % |
| B'. garment crops letterboxed to a square | 67.2 % | 79.0 % | 53.1 % | 77.4 % |
| C. garment crops + new ranking, product = closest photo | 79.5 % | 91.7 % | 62.5 % | 83.9 % |
| **C. garment crops + new ranking, product = mean of 2 closest (default)** | **80.3 %** | **95.2 %** | 59.4 % | 85.5 % |

Read with care:
- **Shoot bias:** a product's photos come from one shoot (same model, same background), so whole photos find their
  siblings through the shoot as much as through the garment. This flatters A. Video frames never share the shoot,
  which is where garment crops matter.
- **Trousers** (the outfit-confusion case) gain the most: +9 points top-1.
- **Sampling noise:** with 229 queries one point is ~2 queries, so A vs B vs C overall is within noise. The clear
  results are letterbox (worse) and top-2 aggregation (+3.5 top-3).

**2. Real OWLv2 detections** (47 clothing / accessory objects from the analysed videos, same 250-product catalogue):
- **Colour agreement** (same or close colour in the top 3, colour from the name, else from the garment crop):
  45 → 47 of 72 slots.
- **Lists changed substantially:** on average 1.4 of the 3 candidates are shared, and the first candidate changed
  for 9 of 47 objects.
- **Visual side-by-side review** (AI-judged, 24 objects; contact sheets in `data/cache/matching_eval/`, not in Git):
  - garment crops better for 3 (leather jacket detected as "suit", black trousers, black jacket);
  - worse for 1 (open shirts cropped to the tee underneath; after the layering rule it is about the same);
  - same or no better for the rest. The 250-product subset has few jackets or shirts close to what is on screen.

```bash
python -m sceneseen match-catalogue --build-index --mode all     # both indexes
python scripts/product_matching_eval.py --queries 600 --sheets 48
```

## Known limits

- **Exact-product accuracy is not measured.** The numbers above prove that candidates are of a compatible type,
  not that they are the right product. Measuring that needs a few videos where someone marks the real product.
- Garment crops are boxes, not masks: a crop of a tee still shows a bit of the jacket over it. Layered outfits
  remain the main crop failure (handled for open shirts only); a tee sold under an open jacket is cropped
  to the tee, which is correct, but a jacket sold over a hoodie includes the hoodie.
- 95 of 1,222 photos keep the full photo (unknown product type, or no confident box: candles, towels, some crochet).
- The evaluation used 250 of 1,183 products; on the full catalogue the numbers will differ (more look-alikes, so lower).
- The detector has no hoodie / shorts / socks / skirt classes: a hoodie is detected as a jacket or shirt, shorts as
  trousers, so those types are only reachable through the broad tier.
- Colour comes from one representative frame and the product name; lighting and printed designs make it
  unreliable, which is why it only nudges the ranking.
- One crop (the object's best detection) is used per object; occurrences in other shots are not averaged.
- The taxonomy was built from 250 of the 1,183 IYS products (the store's public listing rate-limited further
  requests). Run `--taxonomy-report` on the full catalogue to see any product type it does not know yet.
