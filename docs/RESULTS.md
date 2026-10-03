# SceneSeen Phase 1 — Evaluation record

## Evaluation 1 — 2026-10-02 — first 6 human-labelled videos

### Data
| video | language / type | length | shots | true scenes | how labelled |
|---|---|---|---|---|---|
| Spider-Man: No Way Home (first 10 min) | EN film | 9:48 | 156 | 10 | edited model predictions (3 merges, 1 split) |
| Sabe3 Gar ep. 14 | AR series | 32:34 | 116 | 21 | edited model predictions (4 merges) |
| video_01 | EN film | 18:08 | 233 | 7 | edited model predictions (2 merges, 2 splits) |
| video_02 (Pursuit of Happyness clip compilation) | EN, YouTube compilation | 10:52 | 128 | 4 | **from blank** |
| Kamel El Adad | AR series | 8:33 | 152 | 8 | edited model predictions (1 merge) |
| Samir w Shahir w Bahir | AR film | 8:32 | 66 | 11 | **from blank** |

All 6 labels were validated: valid JSON, sorted, inside the video duration, and matched to their video by
content hash. **None were modified.**

**Anchoring caveat.** 4 of 6 labels were made by correcting SceneSeen's predictions. On those, recall is likely
optimistic (a miss only counts if the labeller noticed and added it), and boundary times coincide with the model's
cuts (timing error 0.00 s). The two videos labelled from blank are the least biased evidence. All 6 are therefore
**dev** data (`ground_truth/splits.json`). None can serve as an unbiased final test.

Two environment problems found and fixed during evaluation (they don't affect scores):
- `video_01` and `video_02` in `data/videos/` (and one upload) were iCloud "dataless" placeholders. Reading
  them blocked indefinitely. Evaluation now falls back to the byte-identical local uploaded copy, and `probe`
  reports cloud-only files clearly instead of hanging.

### BASELINE PERFORMANCE (default config, nothing tuned)

| method | F1@1s | F1@2s | F1@3s | P@2s | R@2s | coverage | overflow | F_CO | pred/true scenes |
|---|---|---|---|---|---|---|---|---|---|
| every shot cut = scene | 0.127 | 0.138 | 0.138 | 0.063 | 0.964 | 0.278 | 0.010 | 0.414 | 18.57 |
| adjacent-shot similarity | 0.282 | 0.316 | 0.344 | 0.235 | 0.636 | 0.561 | 0.300 | 0.600 | 3.06 |
| **SceneSeen** | **0.677** | **0.677** | **0.694** | **0.789** | **0.818** | 0.942 | 0.301 | **0.728** | 0.96 |

F1 is macro-averaged over videos; P/R are pooled. Coverage < 1 means over-segmentation; overflow > 0 means
under-segmentation.

Per video (SceneSeen, ±2 s unless stated):

| video | true / pred scenes | P | R | F1@1s | F1@2s | F1@3s | coverage | overflow |
|---|---|---|---|---|---|---|---|---|
| Spider-Man | 10 / 12 | 0.73 | 0.89 | 0.80 | 0.80 | 0.90 | 0.97 | 0.01 |
| Sabe3 Gar | 21 / 25 | 0.83 | 1.00 | 0.91 | 0.91 | 0.91 | 0.92 | 0.00 |
| video_01 | 7 / 7 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.88 | 0.37 |
| video_02 (compilation) | 4 / 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.97 | 1.00 |
| Kamel El Adad | 8 / 9 | 0.88 | 1.00 | 0.93 | 0.93 | 0.93 | 0.98 | 0.00 |
| Samir w Shahir | 11 / 8 | 0.71 | 0.50 | 0.59 | 0.59 | 0.59 | 0.92 | 0.43 |

F1@2s per video for the baselines — every cut: 0.11 / 0.30 / 0.05 / 0.02 / 0.09 / 0.27; adjacent: 0.38 / 0.55 /
0.12 / 0.00 / 0.50 / 0.35. SceneSeen beats both baselines on every video except video_02, where all three fail.

Macro F1@2s: **0.87 on the 4 prediction-edited labels, 0.29 on the 2 independent labels** (0.59 on the
independent film, 0.00 on the compilation).

### Error analysis (12 false boundaries, 10 missed; `scripts/error_analysis.py`, strips in `reports/errors/`)

| pattern | count | evidence |
|---|---|---|
| Same home/office, different room → false split | 4 FP | Sabe3 Gar 107.6 / 206.1 / 738.6 / 925.6 s: hallway → bedroom, living room → bedroom, reception → office |
| Insert / cutaway / exterior shot → false split | 3–4 FP | Spider-Man 553.8 (prop close-up), 301.2 (aerial exterior of the same building); Kamel 438.9 (water close-up on the boat); video_01 719.6 |
| High-score cut rejected by depth test (scenes of 1–3 shots) | 4 FN | Samir 399.5 (score 0.71), 471.4 (0.72), 92.7 (0.47); Spider-Man 557.7 (0.71, before an establishing shot) |
| Graphic wipe not detected by the shot detector | 3 FN | video_02: YouTube ">>" wipe + caption card; scenes fused into one "shot" (nearest cut up to 10.9 s away) |
| Dark → dark location change | 2 FN | video_01 845 s (night office → night car, score 0.29); Samir 232 s (0.448, just below 0.45) |
| Lighting change within a scene | 1 FP | Samir 434 s |
| Label placed 1.0–1.6 s after the cut | 3 labels | Spider-Man 559.4, video_01 845.2, Samir 232.0 (pause-reaction lag; the snap reached only 1 s) |

Not observed: **no dialogue scene was split** (A/B coverage held on all 6 videos), **no fade/dissolve errors**,
and **no fast-cut failures** (errors occur at 0.5–6 cuts per 10 s, i.e. slow editing).

### CHANGES MADE

| # | change | evidence | validation | kept? |
|---|---|---|---|---|
| 1 | Threshold tuning (8,100-config grid, defined before this evaluation) | — | all-6 score 0.735, but **leave-one-video-out 0.606 < defaults 0.677**; hurt 4/6 videos held-out | **Rejected** — defaults kept |
| 2 | `depth_mode = "window"`: prominence vs. the lowest cut within ±window/2 cuts instead of TextTiling's climb (fixes the "adjacent boundaries" misses) | 4 FN above | default thresholds: F1@2s 0.663 (recall 0.818→0.836, precision 0.789→0.719); added to the grid it was never selected, and LOVO stayed 0.606 | **Rejected** as default; kept as an opt-in option for future data |
| 3 | Label snapping: snap to a cut up to 2 s *before* the click / 0.5 s after (was ±1 s) | 3 labels 1.0–1.6 s late | unit tests; affects future labels only | Kept (tooling) |
| 4 | iCloud placeholder handling (`probe` error, local-copy fallback, `-nostdin`) | 3 files hung | evaluation ran on all 6 | Kept (robustness) |
| 5 | `scripts/error_analysis.py` (per-error diagnosis + frame strips) | — | — | Kept (tooling) |

### PERFORMANCE AFTER CHANGES
Unchanged by design. The segmentation algorithm and default thresholds are the same, and re-running `evaluate`
after all code changes reproduces F1@2s **0.677** exactly. No tuned number is reported as accuracy.

### Honest reading
* SceneSeen clearly adds value beyond shot detection: F1@2s 0.68 vs 0.14 (every cut) and 0.32 (adjacent only).
  Scene counts are right on average (0.96×).
* The least biased estimate for real film content is the one film labelled from blank: **F1@2s ≈ 0.6**. The
  0.87 on edited labels is an upper bound.
* With 6 videos (2 unbiased), threshold changes can't be validated: every tuned variant overfit. The
  remaining errors are mostly **semantic** (same apartment, different room; cutaways), and thresholds cannot fix
  those without breaking other videos.


---

## Evaluation 2 — 2026-10-03 — Unique Shots + learning from ground truth

### Data
Three new labels were added and assigned to a new held-out **validation** role (never trained or tuned on):

| video | length | shots | true scenes | how labelled |
|---|---|---|---|---|
| Kamel El Adad +1 (first match) | 2:56 | 52 | 3 | **from blank** |
| Nelly & Sherihan (car) | 10:19 | 112 | 6 | edited model predictions (3 merges) |
| Nelly & Sherihan (stolen car) | 12:02 | 117 | 4 | edited model predictions (3 merges) |

The 6 videos of Evaluation 1 remain **dev**. There is still no final-test set. Provenance of all 9 labels:
`ground_truth/provenance.json` (3 from blank, 6 edited).

### BASELINE (before any change in this round; default config)

| split | F1@1s | F1@2s | F1@3s | P@2s | R@2s | coverage | overflow | pred/true scenes |
|---|---|---|---|---|---|---|---|---|
| dev (6) | 0.677 | 0.677 | 0.694 | 0.789 | 0.818 | 0.942 | 0.301 | 0.96 |
| val (3) | 0.812 | 0.812 | 0.812 | 0.625 | 1.000 | 0.930 | 0.000 | 1.42 |

P/R are pooled over videos. Per video on val: 1.00 (3→3 scenes), 0.77 (6→9), 0.67 (4→7). The rule over-segments the
two Nelly & Sherihan clips.

### CHANGES MADE

| # | change | effect on scene segmentation |
|---|---|---|
| 1 | **Unique Shots** post-processing layer (`unique.py`, API, UI, CLI) | none (verified) |
| 2 | Optional **learned boundary classifier** + `python -m sceneseen train` (`learning.py`) | none: off by default, and the trained model was rejected |
| 3 | Dataset roles dev / val / test; unassigned labels are ignored; `train` refuses val/test leakage | none |
| 4 | Labels now store their provenance and correction log | none |
| 5 | Evaluation can run from cached analysis when a video file is unavailable (iCloud-evicted) | none |

### Learned vs current (macro over videos; every number is for videos the method did not see)

**Dev, leave-one-video-out**

| method | P@2s | R@2s | F1@1s | F1@2s | F1@3s | mean scene-count error | coverage | overflow |
|---|---|---|---|---|---|---|---|---|
| A current rule | 0.664 | 0.704 | 0.677 | **0.677** | 0.694 | 2.00 | 0.942 | 0.301 |
| B tuned rule | 0.580 | 0.656 | 0.588 | 0.606 | 0.626 | 2.50 | 0.885 | 0.274 |
| C logistic regression | 0.558 | 0.652 | 0.580 | 0.580 | 0.598 | 4.50 | 0.877 | 0.355 |
| C gradient boosting | 0.803 | 0.684 | 0.595 | 0.698 | 0.715 | 2.67 | 0.929 | 0.370 |
| D hybrid | 0.654 | 0.704 | 0.671 | 0.671 | 0.687 | 2.17 | 0.934 | 0.305 |

| video (true → predicted scenes) | A rule | B tuned | C logreg | C gbdt | D hybrid |
|---|---|---|---|---|---|
| Spider-Man | 0.80 (10→12) | 0.71 (10→9) | 0.74 (10→11) | 0.80 (10→12) | 0.76 (10→13) |
| Sabe3 Gar | 0.91 (21→25) | 0.89 (21→26) | 0.71 (21→37) | 0.86 (21→25) | 0.91 (21→25) |
| video_01 | 0.83 (7→7) | 0.50 (7→11) | 0.44 (7→4) | 0.44 (7→4) | 0.83 (7→7) |
| video_02 *(blank)* | 0.00 (4→2) | 0.00 (4→3) | 0.00 (4→1) | 0.50 (4→2) | 0.00 (4→2) |
| Kamel El Adad | 0.93 (8→9) | 0.88 (8→10) | 0.82 (8→11) | 0.88 (8→10) | 0.93 (8→9) |
| Samir w Shahir *(blank)* | 0.59 (11→8) | 0.67 (11→9) | 0.76 (11→12) | 0.71 (11→8) | 0.59 (11→8) |

**Held-out validation** (fitted and tuned on the 6 dev videos only)

| method | P@2s | R@2s | F1@1s | F1@2s | F1@3s | mean scene-count error | coverage | overflow |
|---|---|---|---|---|---|---|---|---|
| A current rule | 0.708 | 1.000 | 0.812 | **0.812** | 0.812 | 2.00 | 0.930 | 0.000 |
| B tuned rule | 0.724 | 0.767 | 0.694 | 0.694 | 0.694 | 1.67 | 0.886 | 0.073 |
| C logistic regression | 0.606 | 0.933 | 0.706 | 0.706 | 0.706 | 3.00 | 0.846 | 0.006 |
| C gradient boosting | 0.724 | 0.767 | 0.694 | 0.694 | 0.694 | 1.67 | 0.930 | 0.056 |
| D hybrid | 0.708 | 1.000 | 0.812 | 0.812 | 0.812 | 2.00 | 0.930 | 0.000 |

| video (true → predicted scenes) | A rule | B tuned | C logreg | C gbdt | D hybrid |
|---|---|---|---|---|---|
| Kamel El Adad +1 *(blank)* | 1.00 (3→3) | 0.67 (3→2) | 1.00 (3→3) | 0.67 (3→2) | 1.00 (3→3) |
| Nelly & Sherihan (car) | 0.77 (6→9) | 0.67 (6→8) | 0.57 (6→10) | 0.67 (6→8) | 0.77 (6→9) |
| Nelly & Sherihan (stolen car) | 0.67 (4→7) | 0.75 (4→6) | 0.55 (4→9) | 0.75 (4→6) | 0.67 (4→7) |

**Training fit (in-sample, NOT accuracy):** rule 0.677, tuned 0.735, logistic 0.716, boosting 0.770, hybrid 0.677.

**Decision: all rejected; the current rule remains the default.** B: worse on 6 videos, better on 2. C logistic:
worse on 6, better on 1. C boosting: +0.02 on dev, −0.12 on validation, worse on 5, better on 3. D hybrid: its
selection procedure chose to leave the rule unchanged.

Anchoring caveat: 6 of 9 labels come from edited rule predictions and favour the rule. On the 3 from-blank videos
the mean F1 is rule 0.53, logistic 0.59, boosting 0.63. That is too few videos to conclude anything, but it is
the reason the learned path is kept (off by default) instead of removed.

### PERFORMANCE AFTER CHANGES
Identical to the baseline of this round: dev F1@2s **0.677**, validation **0.812** (re-run after all code changes).
Unique Shots has zero effect on segmentation, and the learned model is not enabled.

### Unique Shots
1,132 shots → 664 unique (41 % reduction) across the 9 videos; per video 21–64 %. Method, calibration and
per-video table: docs/UNIQUE_SHOTS.md.
