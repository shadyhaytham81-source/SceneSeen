# SceneSeen — Project Status

_Last updated: 2026-10-05 · Phase 1 done · Phase 2A built · **Phase 2B (catalogue, product matching, human verification) built, not merged** · Phase 3+ not started_

## What SceneSeen is
A graduation project. The long-term vision is to understand film and TV content scene by scene, so that products,
brands and places on screen can eventually be identified and connected to brands, stylists, set decorators and
viewers.

**Current scope is Phase 1 only:** given a video, find its narrative **scenes**, not just its camera cuts, and
return timestamps, previews, JSON and per-scene clips. A conversation filmed as A → B → A → B is four shots but
one scene.

## Phase 1 architecture
```
video ─► shot detection ─► feature extraction ─► scene grouping ─► scenes (JSON, previews, clips)
          TransNetV2         3 frames / shot        windowed coherence
                             CLIP + colour          + depth test + min length
```
- **Shot detection:** TransNetV2, a pretrained neural shot-boundary detector run on CPU. It catches hard cuts and
  dissolves/fades. Shots shorter than 0.4 s are merged. PySceneDetect is a faster config option.
- **Feature extraction:** 3 frames per shot (at 15/50/85 % of the shot). Each frame gets an **OpenCLIP
  ViT-B/32** embedding (what the place and people look like) and an HSV colour histogram (lighting / palette).
  The results are cached per video.
- **Scene grouping:** for every cut, the shots in a window **before** it are compared with the shots
  **after** it (up to 6 shots, 45 s). If a set-up returns (A after B), it links across the cut, so dialogues stay
  together. A cut becomes a scene boundary when its change score is high (≥ 0.45) and it is a clear local peak
  (TextTiling depth ≥ 0.35), or it is very high (≥ 0.75). Scenes must be ≥ 6 s. It runs in milliseconds from
  cache. No model is trained.
- **Ground truth and evaluation:** human labels in `ground_truth/*.json` (start time of each scene). Metrics:
  precision/recall/F1 at ±1/2/3 s, timing error, coverage/overflow (over/under-segmentation). Every result
  is compared with two baselines. There is a frozen dev/test split and a leave-one-video-out check for any tuning.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · protocol and labelling rules:
[docs/EVALUATION.md](docs/EVALUATION.md) · full results: [docs/RESULTS.md](docs/RESULTS.md).

## Current results (6 videos, default configuration, nothing tuned)
| method | F1@±2s | precision | recall | predicted / true scenes |
|---|---|---|---|---|
| every shot cut = scene (baseline) | 0.14 | 0.06 | 0.96 | 18.6× |
| adjacent-shot similarity (baseline) | 0.32 | 0.24 | 0.64 | 3.1× |
| **SceneSeen** | **0.677** | **0.79** | **0.82** | **0.96×** |

### Which labels are prediction-anchored
4 of the 6 labels were made by **editing SceneSeen's predictions**: Spider-Man, Sabe3 Gar, video_01 and Kamel
El Adad. On these, scores are optimistic (F1 0.87), because a miss only counts if the labeller noticed it. Only
**video_02** and **Samir w Shahir** were labelled **from blank** (F1 0.00 and 0.59). **Least-biased estimate for
real films: about 0.6.**

### Tuning attempted and rejected
- 8,100-setting threshold grid: 0.735 on the videos it was tuned on, but **0.606 on held-out videos**, below the
  untuned 0.677. **Rejected.**
- Alternative depth test ("window" mode) to catch short scenes: F1 fell to 0.663, and it was never chosen
  under leave-one-out. **Rejected** (kept only as an opt-in option).
- The default settings are unchanged from before any real data was seen.

## Known model weaknesses
- Same apartment/office, different room → false new scene.
- Insert/cutaway shots (prop close-ups, exterior of the same building) → false splits.
- Scenes of only 1–3 shots in a row can be missed (depth test).
- Graphic wipes in YouTube compilations aren't seen as transitions → scenes merged.
- Dark-to-dark location changes (night office → night car) can be merged.
- Visual only: no audio/dialogue continuity, no character identity.
- Shot detection ≈ 0.15× video duration on CPU (the slowest stage).

What works well: no dialogue scene was split in any of the 6 videos. No errors at fades, dissolves or fast cuts.
Scene counts are right on average.

## Added on 2026-10-03 (merged into `main`)
Default behaviour: **rule-based segmentation** + **Unique Shots**. The learned boundary classifier is kept but
**disabled** (`[boundary_model] path = ""`); it is only used if a model file is explicitly configured.

- **Unique Shots:** inside each scene, shots from the same camera set-up are grouped under one representative
  (nothing deleted). Similarity = CLIP multi-frame cosine + colour + perceptual hash, calibrated on 177 labelled
  shot pairs. 1,132 shots → 664 unique (41 %) on the 9 labelled videos. It is post-processing only, with zero
  effect on segmentation. See [docs/UNIQUE_SHOTS.md](docs/UNIQUE_SHOTS.md).
- **Learning from ground truth:** `python -m sceneseen train` builds one example per shot cut from the labels and
  trains a small boundary classifier on the frozen signals. **First result: rejected.** Held-out validation F1 is
  0.706 (logistic) and 0.694 (boosting) against 0.812 for the current rule. The rule stays the default; the
  learned path is off unless a model is explicitly configured. See [docs/LEARNING.md](docs/LEARNING.md).
- 3 new labelled videos form a held-out **validation** set (current rule: F1@±2s 0.812).
- Dataset roles are now explicit (dev / val / test). Unassigned labels are never used.

## Dataset structure
```
ground_truth/<name>.json           human labels (in Git)        {"video": ..., "boundaries": [seconds...], "provenance": {...}}
ground_truth/splits.json           roles (in Git)               dev = 6 videos (training/development), val = 3, test = none yet
ground_truth/provenance.json       how each existing label was made (blank vs edited predictions) + correction logs
ground_truth/videos_manifest.json  exact name/size/SHA-256 of the 9 labelled videos (in Git)
ground_truth/unique_shots/         shot-pair labels used to calibrate Unique Shots
models/                            versioned trained-model artifacts (JSON; none is enabled)
data/videos/<exact name>           the labelled videos (NOT in Git, shared separately)
data/uploads, cache, exports       created by the app (NOT in Git)
```
Collaborators: put the shared videos in `data/videos/`, run `python -m sceneseen check-data --verify`.
New videos are simply uploaded through the website.

## What remains before Phase 1 can be frozen
1. **Label more videos from blank** (Start blank). Only 3 of 9 labels are independent, and that is the bottleneck
   for both honest accuracy and learning. Target: at least 4 more for `val` and a separate 4–6 for `test`.
2. Re-run `python -m sceneseen train`. Accept a learned model **only** if it beats the rule on validation.
3. **Freeze** the model/config (commit + tag).
4. Evaluate **once** on the unseen `test` videos: `python -m sceneseen evaluate --split test`. No tuning afterwards.

Practical note: keep the project **outside iCloud-synced folders** (Desktop/Documents with "Optimise Mac Storage").
macOS evicts videos, caches and even virtualenv files there, which makes reads hang. If it must stay there, name
the environment `.venv.nosync` (iCloud ignores `*.nosync`).

## Phase 2A — Commercial Scene Understanding (2026-10-05, branch `feature/commercial-scene-understanding`)
Built on top of Phase 1 without changing it (dev F1 0.677 / validation 0.812 identical before and after).

- **What it does:** per scene, commercially relevant objects (38 types in 7 categories) with confidence, relevance,
  crop and occurrence list, plus scene context (venue, indoor/outdoor). No brand or product identification.
- **Model:** OWLv2 base (Apache-2.0), pretrained, nothing trained. Chosen over Grounding DINO (7× slower here,
  merged labels), Florence-2 (generic labels, no confidence scores) and OmDet-Turbo (fast, ~50 % precise).
- **Speed:** one frame per unique shot; ~0.9 s per unique shot on Apple MPS (1.5 s on CPU); a 12-minute clip takes
  about 40 s; cached results return in under 0.3 s.
- **Precision:** 84.9 % of displayed objects correct on 5 held-out videos (118 / 139; AI-reviewed, not yet
  human-verified). Scene-context venues ≈ 70 %.
- **Review tool:** Review switch → seven verdicts + Missed on each object; `python -m sceneseen commercial-report`.
- Details, measurements and known failure cases: [docs/COMMERCIAL.md](docs/COMMERCIAL.md).

Next for Phase 2A: have a human confirm a sample of detections in the review tool (this replaces the AI review as
the precision figure), and record missed objects to see what recall is losing.

## Phase 2B — Catalogue, Product Matching, Verification (2026-10-05, branch `feature/product-catalog-matching`)
Built on top of Phase 2A without changing Phase 1 or the detector (dev F1 0.677 / validation 0.812 unchanged).
**Not merged into `main`; waiting for an independent audit.**

```
Video → Scenes → Unique Shots → Commercial Objects → Catalogue → Ranked candidates → Confidence → Human
verification → Confirmed products per scene
```
- **Catalogue:** brands, products, variants, several images per product (front/back/side/detail/lifestyle),
  category, object type, SKU, external ID, URL, price, availability. SQLite through SQLAlchemy (one local file);
  the same code runs on PostgreSQL, pgvector is the documented next step. Catalog page with search, filters,
  paging, multi-image upload, archive. [docs/CATALOG.md](docs/CATALOG.md)
- **Matching:** OpenCLIP embedding (the model Phase 1 already has) + colour histogram, category gating, best
  reference image, mean over up to 5 occurrences. Marqo e-commerce embeddings ranked better in the study and are
  an optional profile (`config/matching_marqo.toml`). Nothing trained.
- **Confidence:** calibrated from the score *and* its lead over the next product. Held-out: "high confidence" is
  right 93 % of the time (41/44); a product that is not in the catalogue is shown as high confidence in 2 % of
  cases; the right product is ranked first in 67 % and is high-confidence in 27 %. Everything else is "possible",
  "uncertain" or "unknown product" and goes to a person. Detection confidence, commercial relevance, match
  confidence and verification status stay separate fields.
- **Human verification:** Confirm / No match / Search catalog / Change, with an append-only audit trail (who,
  when, product, previous decision, AI confidence and model versions at that moment). The AI never confirms.
- **Review verdicts (Phase 2A):** Correct, Wrong, Unsure, Wrong label (+ the right label), Not useful, Unclear
  image, Duplicate. Stored as structured data; nothing is retrained.
- **Speed:** exact vector search, 0.6 ms per object against 10,000 products / 30,000 images; first match of a
  video 0.2–2 s after the model is loaded; cached afterwards. Runs as a background job with progress.
- **Failure isolation:** catalogue down, model missing, matcher error → objects are still listed, scenes untouched.
- **Honest limits:** calibration labels are AI-made (62 objects), references in the study are film crops, and
  **no real product catalogue has been tested yet**. [docs/MATCHING.md](docs/MATCHING.md)

Needs a human: add 20–50 real products with studio photos that appear in a labelled video and check the ranking
and the confidence states; confirm or correct a sample of detections with the new verdicts; try the Catalog page
with real data (long names, many images, bulk entry speed).

## ⛔ Not started — and must not start yet
Placement Opportunity Engine, QR codes / Shop-the-Episode, viewer analytics, brand campaign analytics, screen
recognition and engagement prediction. Phase 2 ends at confirmed products per scene (`products.json`), which is
the input those phases need. None of them is implemented.
