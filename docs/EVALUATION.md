# SceneSeen — Evaluation methodology & labelling guide

## What is measured

Predicted scene boundaries (the start time of every scene except the first) are compared with human labels.

| Metric | Meaning |
|---|---|
| **Precision / Recall / F1 @ ±1 s, ±2 s, ±3 s** | Predictions and labels are matched **one-to-one** (optimal assignment, so one prediction can't satisfy two labels) within the tolerance. Headline number: **macro F1 @ ±2 s** (average over videos). |
| **Timing error** | Mean \|predicted − true\| of matched boundaries. |
| **Predicted / true scene count** | > 1 = over-segmentation, < 1 = under-segmentation. |
| **Coverage / Overflow / F_CO** | Vendrig & Worring 2002, standard in scene segmentation. Coverage drops when true scenes are split (over-segmentation). Overflow rises when predicted scenes spill into neighbouring true scenes (under-segmentation). F_CO is their harmonic mean. Unlike F1, it doesn't depend on a tolerance. |

Why ±2 s: human labellers disagree by a second or two on where a scene "starts" (an establishing shot, a sound
bridge). ±1 s is reported to show timing precision, and ±3 s to show whether misses are near-misses.

## Baselines (always reported next to SceneSeen)

* **baseline_all_shots**: every shot cut is a scene boundary. Shows what shot detection alone gives.
* **baseline_adjacent**: same features and selection, but each cut only compares the two neighbouring shots (no
  context window). It isolates the value of the windowed-coherence idea that handles A/B dialogue.

SceneSeen adds value only if it beats both.

## Protocol (no data leakage)

1. Every labelled video has one role in `ground_truth/splits.json`: **dev** (training/development), **val**
   (held-out validation) or **test** (final test). A newly saved label has no role and is ignored by `evaluate`,
   `tune` and `train` until you add its name to one of the lists, so nothing leaks by accident. The file is
   never rewritten by the tools.
2. `python -m sceneseen tune` grid-searches 8,100 grouping configurations **on dev only**. It also reports a
   **leave-one-video-out** score: for each dev video, tune on the others and score the held-out one. If that is
   below the untuned defaults, tuning is overfitting. The command says so and you should keep the defaults.
3. `python -m sceneseen --config config/tuned.toml evaluate --split test` is run **once**, at the end. Do not
   tune again after seeing test numbers. If you must iterate, label new videos for dev.

Shot detection and CLIP are pretrained and not tuned on our data. Only the grouping thresholds are tuned.

## Labelling guide (what counts as a scene boundary)

A **scene** is a continuous unit of story in one place and time. Start a new scene when:
- the **location** changes (restaurant → street, even if the characters stay), or
- **time jumps** (same place, next morning; a flashback starts or ends), or
- the story moves to a **different group of characters elsewhere** (intercutting between two places = new scene
  at every switch, if each segment is at least a few seconds).

Do **not** start a new scene for:
- camera cuts within the same conversation (A/B close-ups, reverse shots, inserts of hands/objects),
- moving between rooms in one continuous action (follow-shot through a house), unless the story clearly resets,
- title cards. Put a boundary *after* opening titles if the story starts there; ignore mid-film captions.

Put the boundary at the **first frame of the new scene**: the first cut into the new location/time. If a
dissolve links two scenes, use the middle of the dissolve. Precision to about 0.5 s is enough.

Ambiguous cases (montages, dream sequences): decide once, write the rule in `notes`, and apply it consistently.

### How to label: two options

**A. In the SceneSeen UI (recommended)**
1. `python -m sceneseen serve`, tick **Developer**, upload the video and analyse it.
2. Turn on **Edit scenes**. Click **Start blank** to label without being influenced by SceneSeen's predictions.
   This is recommended for the **test** videos.
3. Watch the video. At each scene start, pause and click **Split at playhead**. It snaps to the nearest real cut
   up to 2 s *before* the playhead (people pause late) or 0.5 s after. Use **Merge with next** to remove a wrong
   boundary.
4. Click **Save as ground truth** and enter your name. This writes `ground_truth/<video>.json` and places the
   video in `data/videos/`.

Starting from SceneSeen's predictions and correcting them is faster, but it can bias labels towards the model
(anchoring). That's acceptable for **dev** videos. For **test** videos use **Start blank**. In the first
evaluation, the gap was large: F1 0.87 on edited labels vs 0.29 on labels made from blank (docs/RESULTS.md).

Prefer continuous film/episode segments. YouTube *compilations* join clips with graphic wipes and caption cards,
which the shot detector doesn't recognise as transitions.

**B. By hand.** Write `ground_truth/<video_stem>.json`:

```json
{
  "video": "video_01.mp4",
  "boundaries": [48.2, 127.4, 221.8],
  "annotator": "shady",
  "notes": "flashback at 3:41 counted as its own scene"
}
```

`boundaries` are seconds, strictly increasing, > 0 and < duration. The first scene starts at 0 implicitly. The
video file must be at `data/videos/<video>`.

## Machinery check (synthetic, NOT a benchmark)

`scripts/make_synthetic.py --set 4` concatenates runs of consecutive *Tears of Steel* shots in new orders, so the
boundaries are known by construction (`config/synthetic_check.toml` points evaluate/tune at them). This checks that
the code works end to end. It does **not** measure real-film performance, because synthetic joins are cleaner
than real scene changes.

Results of the check (default thresholds, 3 dev videos):

| method | F1@2s | pred/true scenes |
|---|---|---|
| baseline_all_shots | 0.129 | 11.0 |
| baseline_adjacent | 0.367 | 2.4 |
| sceneseen | 0.802 | 1.19 |

Tuning on the 3 synthetic dev videos reached 0.95 on those same videos, but only 0.70 leave-one-out (below the
untuned 0.80). With very few videos, tuning overfits. That is why the leave-one-out check is part of `tune`.
