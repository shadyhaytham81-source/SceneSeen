# SceneSeen — Project Status

_Last updated: 2026-10-02_

## What SceneSeen is
A graduation project. Long-term goal: understand film/TV content scene by scene, so that products, brands and
places on screen can eventually be identified and connected to brands, stylists, set decorators and viewers.

**Current scope is Phase 1 only:** video → shots → narrative **scenes** (timestamps, previews, JSON, clips).

## Phase 1 architecture
```
video ─► shot detection ─► 3 frames/shot ─► CLIP + colour features ─► scene grouping ─► scenes
         TransNetV2         (15/50/85 %)      OpenCLIP ViT-B/32        windowed coherence
                                                                       + TextTiling depth
                                                                       + min scene length
```
Shot detection and features are cached per video, so re-grouping takes milliseconds. Everything runs locally:
no paid APIs and no model training. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Completed
- Pipeline: ingestion, shot detection, features, grouping, cache (`sceneseen/`)
- Web app: upload, progress, scene cards and timeline, preview, export JSON and clips, developer/debug view,
  split/merge corrections, save-as-ground-truth (`server/`)
- CLI: `analyze`, `evaluate`, `tune`, `check-data`, `serve`
- Evaluation framework: P/R/F1 at ±1/2/3 s, timing error, coverage/overflow, two baselines, frozen dev/test split,
  leave-one-video-out tuning check (`docs/EVALUATION.md`)
- Error-analysis tool (`scripts/error_analysis.py`)
- 6 human-labelled videos (`ground_truth/`); first real evaluation (`docs/RESULTS.md`)
- 41 automated tests (`python -m pytest`; `-m slow` for end-to-end)

## Current evaluation results (6 videos, default config, nothing tuned)
| method | F1@2s | precision | recall | predicted / true scenes |
|---|---|---|---|---|
| every shot cut = scene | 0.14 | 0.06 | 0.96 | 18.6× |
| adjacent-shot similarity | 0.32 | 0.24 | 0.64 | 3.1× |
| **SceneSeen** | **0.68** | **0.79** | **0.82** | **0.96×** |

- 4 of 6 labels were made by editing SceneSeen's predictions, so they are optimistic (F1 0.87). The 2 labelled from
  blank give 0.29 (film: 0.59; YouTube compilation: 0.00). **Least-biased estimate for real films: about 0.6.**
- Threshold tuning was tried and **rejected**: 0.735 on the tuning videos but 0.606 held-out, below the defaults.
- No dialogue scene was ever split. No errors at fades, dissolves or fast cuts.

## Known limitations
- Same apartment or office, different room → false new scene (Sabe3 Gar).
- Insert / cutaway shots (prop close-ups, exteriors of the same building) → false splits.
- Scenes of only 1–3 shots in a row can be missed (TextTiling depth test).
- Graphic wipes in YouTube compilations aren't detected as transitions → scenes merged.
- Dark-to-dark location changes (night interior → night car) can be merged.
- Shot detection is the slowest stage: about 0.15× video duration on CPU.
- Visual-only: no audio or dialogue continuity, no character identity.
- Only 2 of 6 labels are unbiased. **There is no unbiased test set yet.**

## Dataset structure
```
ground_truth/<video name>.json   human labels (in Git)        {"video": ..., "boundaries": [seconds...], ...}
ground_truth/splits.json         dev/test split (in Git)      all 6 current videos are dev
data/videos/<video name>         the videos (NOT in Git)      exact file name from each label's "video" field
data/cache, uploads, exports     generated (NOT in Git)
```
Collaborators: obtain the videos separately, copy them into `data/videos/`, run `python -m sceneseen check-data`.
See [data/README.md](data/README.md).

## What should happen next
1. **Collect a small unseen test set:** 4–6 continuous film/series segments, 5–10 min each, no YouTube
   compilations. Label every one **from blank** (Edit scenes → Start blank) and put them in the `test` list of
   `ground_truth/splits.json`.
2. Run `python -m sceneseen evaluate --split test` **once** with the current defaults. Do not tune afterwards.
3. If test F1@2s ≥ ~0.55: freeze Phase 1. If below ~0.5: improve Phase 1 first (most promising: audio/dialogue
   continuity across cuts, then character continuity), validated with leave-one-video-out.

## ⛔ Phase 2 must NOT start yet
No object detection, clothing/product recognition, commercial-opportunity scoring, catalog matching or viewer
features until Phase 1 has an unbiased test result and a freeze decision.
