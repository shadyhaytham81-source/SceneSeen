# Phase 2B — Product Catalogue Matching

Product catalogue matching connects SceneSeen's object-level understanding (Phase 2A) to actual retail catalogues, starting with **IN YOUR SHOE (IYS)**.

```
Video → Scenes → Unique Shots ─► OWLv2 Detector ─► Commercial Object Candidates (Phase 2A)
                                                                 │
                                                    Cropped Candidate Box
                                                                 │
                                                       OpenCLIP ViT-B/32
                                                                 │
                                                      Query Embedding (512-d)
                                                                 │
                                                   Cosine Similarity Search
                                                                 ▼
Catalogue Persistent Index (Phase 2B) ──────────► Top 3 Visually Similar Products
(embeddings.npz + manifest.json)                   (Images, Brand, Title, Price, URL)
```

---

## Key Principles & Design Decisions

1. **System Separation (Detector vs. Matcher):**
   - **OWLv2** detects commercial objects (e.g. `jacket`, `hat`, `trousers`, `handbag`).
   - **OpenCLIP** (`ViT-B-32`, `laion2b_s34b_b79k`) extracts visual embeddings for both catalogue images and detected object crops.
   - Retrieval is based on **cosine similarity**. Highest-scoring candidates are visual similarity recommendations, not asserted exact ground-truth matches.

2. **Unified Preprocessing & Embeddings:**
   - The exact same OpenCLIP model and transformation pipeline (`preprocess`) are used for catalogue images and object crops.
   - Embeddings are L2-normalized so that cosine similarity is computed via dot products.

3. **Persistent Index & Resumability:**
   - Embeddings are stored per brand in `scraper/output/<brand>/embeddings.npz` (or `data/catalogues/<brand>/`).
   - Metadata is stored in `embeddings_manifest.json`, mapping each embedding vector to `brand`, `product_id`, `handle`, `name`, `category`, `product_type`, `price`, `currency`, `url`, and local/remote image paths.
   - Loading the precomputed index takes **~40 milliseconds** for 5,461 images.
   - Repeated runs do not recompute existing embeddings.

4. **Multi-Brand Extensibility:**
   - `MultiBrandCatalogue` automatically discovers all brand directories in `scraper/output/` and `data/catalogues/`.
   - Adding a new brand (e.g. `scraper/output/another_brand/`) only indexes the new brand; existing brand indices are loaded directly from disk in milliseconds.

5. **Deduplication by Product:**
   - A single product often has multiple photos (angles, colors, lookbook).
   - The matcher computes similarity across all images and selects the best-matching image per product, returning top `k` unique products (default: 3).

---

## Web UI & Workflow

1. **Run SceneSeen as usual:**
   - Drop in any video or select an existing one.
   - When commercial analysis is run or loaded, SceneSeen identifies commercial objects.

2. **Tab Switch in Commercial Objects Section:**
   - **Detected Objects**: Standard OWLv2 detection cards by scene. Clicking any card opens the detail view, which includes a drawer displaying the **Top Visual Matches** from IN YOUR SHOE.
   - **Product Matching**: Dedicated visual comparison view showing each detected object side-by-side with its **Top 3 Candidate Products** (thumbnail, visual match percentage badge, brand, name, price, and store link).

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/videos/{vid}/commercial/matches` | `GET` | Retrieve or compute visual product matches for all displayed candidates in a video. Caches in `data/cache/<vid>/commercial/matches.json`. |
| `/api/catalogue/image` | `GET` | Safely serve downloaded catalogue images with optional thumbnailing (`?rel=...&brand=...&size=320`). |

---

## CLI Usage

```bash
# Build or inspect catalogue index
python -m sceneseen match-catalogue --brand in_your_shoe

# Force re-index / incremental update
python -m sceneseen match-catalogue --brand in_your_shoe --build-index

# Run commercial analysis on a video and print visual product matches
python -m sceneseen commercial data/uploads/92c2cd502301766d.mp4 --match

# Or run the matching module directly
python -m sceneseen.commercial.matching --build-index
```
