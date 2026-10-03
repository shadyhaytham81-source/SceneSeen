# Unique Shots — repeated-shot detection

A post-processing layer on top of Phase 1. Inside every scene, shots that come from the **same camera
set-up** (same framing, angle, background and subject placement) are grouped under one representative shot.

```
Original:  A → B → A → A → C → B          Unique:  A (×3) · B (×2) · C (×1)
```

**Nothing is deleted.** Every original shot stays in the shot list and belongs to exactly one unique-shot group.
Scene segmentation is not touched: evaluation before and after adding this layer is identical (dev F1@±2s 0.677,
validation 0.812).

## Similarity method

Everything is computed from data SceneSeen already caches. CLIP is never re-run.

| signal | what it measures | computed from |
|---|---|---|
| **CLIP multi-frame cosine** (`clip_matched`) | same content: people, objects, setting | cached OpenCLIP embeddings of the 3 sampled frames per shot |
| **Colour** | same light and palette | cached HSV histograms |
| **Perceptual hash** (`phash`) | same framing / spatial layout | the cached mid-shot thumbnail (64-bit DCT hash) |

Cosine similarity of two embeddings `a`, `b`:

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Embeddings are L2-normalised once, so the full frame-to-frame similarity table for a video is a single matrix
product (`E @ E.T`), with no Python loops. For two shots with 3 sampled frames each, `clip_matched` matches every
frame of one shot to its most similar frame in the other, in both directions, and averages. This tolerates actor
movement, expression changes and small camera moves. A shot only scores high if *all* its sampled frames find a
counterpart.

The three signals are combined into one probability:

```
p(duplicate) = sigmoid(2.776·clip_matched + 3.540·colour + 4.089·phash − 7.886)
```

| state | rule | meaning |
|---|---|---|
| **Duplicate** | p ≥ 0.60 | same camera set-up → grouped |
| **Uncertain** | 0.35 ≤ p < 0.60 | reported as "possibly the same set-up", never merged |
| **Different** | p < 0.35 | different shot |

Groups are built per scene with average-linkage clustering, so one borderline pair cannot chain unrelated shots
together.

## Why not CLIP cosine alone? (calibration)

`scripts/unique_shots_study.py` sampled 187 within-scene shot pairs from the 9 labelled videos, evenly across the
CLIP similarity range so the decision boundary is densely covered. They were labelled **blind** (shuffled, no
scores shown) as duplicate / different / unsure. Result: 95 duplicate, 82 different, 10 unsure (excluded).

> The pair labels were made by the AI assistant from mid-shot thumbnails, **not by a human**. They are stored in
> `ground_truth/unique_shots/` so anyone can review or correct them and re-run the study.

| method | ROC-AUC | best F1 | F1 with threshold chosen on other videos |
|---|---|---|---|
| CLIP cosine, mean embedding | 0.971 | 0.921 | 0.911 |
| CLIP cosine, best frame pair | 0.969 | 0.916 | 0.911 |
| CLIP cosine, multi-frame matched | 0.969 | 0.912 | 0.873 |
| Colour histogram | 0.962 | 0.916 | 0.911 |
| Perceptual hash | 0.964 | 0.903 | 0.865 |
| SSIM | 0.953 | 0.925 | 0.914 |
| **CLIP + colour + pHash (used)** | **0.985** | **0.942** | held-out P 0.94 / R 0.95 at p ≥ 0.5 |
| CLIP + colour + pHash + SSIM | 0.985 | 0.957 | 0.952 |

- CLIP cosine alone is good but leaves a wide ambiguous band. For CLIP similarity between 0.84 and 0.91, duplicates
  and different shots are mixed (28 vs 14 in the sample). No labelled duplicate was below 0.804, and no labelled
  different pair at or above 0.915.
- CLIP knows *what* is in the frame; the perceptual hash knows *how it is framed*. Together they separate
  "same actor, different angle" from "same camera set-up".
- SSIM adds about one point of F1, which is inside the noise for 177 pairs. It is the only signal that needs
  per-pair image work, so it is not used.
- Multi-frame vs single mean embedding: statistically indistinguishable here. Multi-frame matching is used because
  it is the more robust definition when a shot contains movement.

**Threshold (0.60), from leave-one-video-out probabilities** (weights fitted on the other 8 videos):

| p ≥ | precision | recall |
|---|---|---|
| 0.4 | 0.86 | 0.97 |
| 0.5 | 0.94 | 0.95 |
| **0.6** | **0.97** | **0.91** |
| 0.7 | 1.00 | 0.80 |

0.60 favours precision: wrongly merging two different shots hides a shot, while missing a duplicate only shows one
extra thumbnail. No true duplicate scored below 0.3 held-out, hence the "different" threshold of 0.35.
`reports/unique_shots/decision_boundary_examples.jpg` (local, not committed) shows the 24 pairs closest to the
threshold for visual inspection. Re-create it and the table with:

```bash
python scripts/unique_shots_study.py evaluate
```

Known weak spot: two-shots vs. singles of the same people in the same light (e.g. pairs p034, p050, p085) can score
0.6–0.7 and be grouped.

## Representative shot

Inside a group, each occurrence is ranked on five criteria and the weighted ranks are summed:

| criterion | weight | why |
|---|---|---|
| centrality (mean similarity to the other occurrences) | 0.35 | the most typical view of the set-up |
| sharpness (Laplacian variance of the mid frame) | 0.25 | avoids motion blur |
| duration | 0.20 | longer occurrences show more |
| stability (agreement of the shot's own sampled frames) | 0.10 | avoids fades, whip-pans, transition frames |
| exposure (distance from mid-grey) | 0.10 | avoids near-black / blown-out frames |

## Output

`GET /api/videos/{id}/unique-shots`, and `unique_shots.json` next to `scenes.json` from `python -m sceneseen analyze`:

```json
{
  "summary": {"original_shots": 117, "unique_shots": 45, "repeated_shots": 72, "reduction": 0.6154},
  "scenes": [{
    "scene_id": 4, "original_shots": 47, "unique_shots": 12, "repeated_shots": 35, "reduction": 0.7447,
    "unique": [{
      "unique_shot_id": "S04-U03",
      "representative_shot_id": 106,
      "shot_ids": [72, 74, 84, 86, 91, 102, 106, 108, 111, 115],
      "duplicate_shot_ids": [72, 74, 84, 86, 91, 102, 108, 111, 115],
      "occurrence_count": 10,
      "occurrences": [{"shot_id": 72, "start": 471.2, "end": 473.6, "similarity_to_representative": 0.79,
                       "clip_cosine_to_representative": 0.93}],
      "possible_duplicates_of": [{"unique_shot_id": "S04-U07", "max_similarity": 0.41}]
    }]
  }]
}
```

## Results on the 9 labelled videos (within human-labelled scenes)

| video | shots | unique | repeated | reduction |
|---|---|---|---|---|
| Spider-Man NWH | 156 | 124 | 32 | 21 % |
| Sabe3 Gar ep. 14 | 116 | 64 | 52 | 45 % |
| video_01 | 233 | 146 | 87 | 37 % |
| video_02 | 128 | 87 | 41 | 32 % |
| Kamel El Adad +1 (match) | 52 | 41 | 11 | 21 % |
| Kamel El Adad | 152 | 69 | 83 | 55 % |
| Nelly & Sherihan (car) | 112 | 40 | 72 | 64 % |
| Nelly & Sherihan (stolen car) | 117 | 45 | 72 | 62 % |
| Samir w Shahir w Bahir | 66 | 48 | 18 | 27 % |
| **Total** | **1132** | **664** | **468** | **41 %** |

Dialogue-heavy series reduce most; action and sports reduce least. Runtime: about 1 ms per shot once for thumbnail
statistics (cached in `visual_<key>.npz`), then 5–20 ms per video for similarity and grouping. Results are cached
per scene layout (`unique_<key>_<hash>.json`).

## Configuration

`[unique_shots]` in `config/default.toml`: weights, both thresholds, linkage and representative weights.
