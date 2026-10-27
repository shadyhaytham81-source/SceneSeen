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
                     4. OpenCLIP embedding of the detected crop
                     5. cosine similarity with every image of the remaining products
                        → best image per product → small colour adjustment
                                        ▼
                     top 3 products: image, name, brand, price, similarity
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

## Rules

- **OWLv2 detects; the taxonomy decides what is comparable; embeddings rank.** Neither replaces the other.
- **Strict first, broader only if needed.** If fewer than 3 products of a strict type exist, the remaining slots are
  filled from the *broad* types (marked "broader category" in the UI, logged as a fallback). Example: IYS sells no
  backpacks, so a detected backpack is compared with tote bags, marked as broader.
- **Never unrelated products.** Object types with no compatible product type (car, lamp, phone, sofa) get no
  candidates, with the reason. Products whose type is unknown are never offered.
- **Colour ranks, it does not filter.** When the detector reports one clear colour and the product name names one,
  the score moves by +0.04 (same colour), +0.02 (close colour), −0.04 (different) or −0.01 (different, multicolour
  product). Otherwise there is no adjustment. A good match in another colour is moved down, never removed.
  These weights are a heuristic, not calibrated: `[catalogue] color_*` in `config/default.toml`.
- **Similarity is not a probability.** It is the cosine similarity of the crop and the product's closest image.
  For these crops good candidates typically score 0.30–0.50 and differ by a few hundredths. The UI shows the number
  without calling it a percentage or a "match".

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
  OpenCLIP preprocessing embed catalogue images and detected crops (crop = detected box + 8 % padding).
- Files per brand (next to `catalogue.jsonl`):

| file | content |
|---|---|
| `embeddings.npz` | float32 `[images, 512]`, L2-normalised; row *i* belongs to manifest record *i* |
| `embeddings_manifest.json` | per row: brand, product id, handle, name, product type, price, currency, product URL, image path (relative to the brand folder), image path for serving, source CDN URL, file size + mtime |
| `embeddings_meta.json` | model, pretrained weights, preprocessing fingerprint, dimension, counts, build time |

- **Nothing is regenerated on a normal run.** Building is a separate command and only embeds what changed:

| change | what happens |
|---|---|
| new images or products | only those are embedded |
| an image file replaced (size or modification time changed) | only that image is re-embedded |
| an image deleted | its row is dropped |
| product name / price / type changed in `catalogue.jsonl` | no re-embedding; the manifest is refreshed on the next build (the taxonomy is applied when loading, not stored in the embeddings) |
| taxonomy edited | no re-embedding; cached match results are recomputed automatically |
| embedding model, weights or preprocessing changed | **everything must be re-embedded**; matching refuses the old embeddings and says so |

- Old indexes made before `embeddings_meta.json` existed (the first version of this matcher) are accepted as
  ViT-B/32 / laion2b, and the next `--build-index` records their version **without re-embedding**.
- Each brand is its own folder with its own index, so adding a brand never rebuilds the others.

```bash
python -m sceneseen match-catalogue                         # status of every brand found
python -m sceneseen match-catalogue --build-index           # embed new / changed images (all brands)
python -m sceneseen match-catalogue --build-index --brand in_your_shoe --force   # re-embed everything
python -m sceneseen match-catalogue --verify 50             # re-embed 50 sampled images: do rows still match their records?
```

On an Apple M4 (MPS) embedding runs at roughly 50 images/s (257 sample images: 10 s including model load); the full IYS catalogue takes a few minutes once.
Loading an existing index takes well under a second.

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

## Known limits

- **Exact-product accuracy is not measured.** The numbers above prove that candidates are of a compatible type,
  not that they are the right product. Measuring that needs a few videos where someone marks the real product.
- Lifestyle photos still contain other garments; the filter removes wrong-type *products*, not the other garments
  inside a right-type product's photo. Preferring product-only ("flat") images would help further.
- The detector has no hoodie / shorts / socks / skirt classes: a hoodie is detected as a jacket or shirt, shorts as
  trousers, so those types are only reachable through the broad tier.
- Colour comes from one representative frame and the product name; lighting and printed designs make it
  unreliable, which is why it only nudges the ranking.
- One crop (the object's best detection) is used per object; occurrences in other shots are not averaged.
- The taxonomy was built from 250 of the 1,183 IYS products (the store's public listing rate-limited further
  requests). Run `--taxonomy-report` on the full catalogue to see any product type it does not know yet.
