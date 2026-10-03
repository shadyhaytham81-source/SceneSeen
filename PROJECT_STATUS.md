# SceneSeen — Project Status

_Last updated: 2026-10-03 · Phase 1 (scene segmentation) · **Phase 2 has NOT started**_

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

## ⛔ Phase 2 has NOT started — and must not start yet
No object detection, clothing/product recognition, commercial-opportunity scoring, catalog matching or viewer
features until Phase 1 is frozen and has its final unseen-test result. Unique Shots is a Phase 1 post-processing
layer (it reduces how many frames a later phase would need to look at); it detects no objects.
