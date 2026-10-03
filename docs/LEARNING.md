# Learning from ground truth

## What is trainable in SceneSeen

| component | kind | trained by us? |
|---|---|---|
| TransNetV2 (shot detection) | pretrained neural network | **No, frozen.** |
| OpenCLIP ViT-B/32 (frame embeddings) | pretrained neural network | **No, frozen.** |
| Scene grouping (`grouping.py`) | hand-designed logic: windowed coherence + depth test | **No weights.** 11 tunable numbers (thresholds, window sizes, weights). |
| Unique Shots combiner | 3 weights + bias | Fitted once on 177 labelled shot pairs (docs/UNIQUE_SHOTS.md). |
| **Boundary classifier (`learning.py`)** | small supervised model on top of the frozen signals | **Yes. This is the only thing `train` trains.** |

So manual corrections cannot "update the model" directly: the current scene grouping has nothing to update except
thresholds. A handful of videos also cannot retrain TransNetV2 or CLIP. The sound way to benefit from ground truth
is a small classifier that learns how to *combine* SceneSeen's existing signals into a boundary decision.

## Ground truth → training examples

One example per **shot cut**. Target: `1` if a human labelled a scene boundary at that cut, else `0`. A label is
assigned to its nearest cut within 2 s, one-to-one. Labels with no cut nearby are "unreachable" (the shot detector
produced no cut there; reported, not learnable).

16 features per cut, all from cached data (`learning.FEATURES`):
- the rule's own signals: boundary score, TextTiling depth, window depth, score relative to nearby cuts;
- the score at two other context sizes (2 and 10 shots);
- CLIP and colour similarity of the two shots at the cut;
- similarity between the 4 shots before and the 4 after, and how coherent each side is internally;
- durations of the shots before and after, and editing pace;
- the shot detector's transition probability and whether the transition is gradual.

Current data: **845 examples from the 6 dev videos (53 boundaries, 792 non-boundaries)**, 278 examples from the 3
validation videos (10 boundaries). 2 dev boundaries are unreachable (graphic wipes in video_02).

## Corrections are stored as data

- Every split/merge is logged in `data/cache/<id>/corrections.json`, together with the prediction it started from.
- **Save as ground truth** now writes a `provenance` block into the label file: `label_method` (`blank` or
  `edited_predictions`), operation counts, the model's original prediction and the full correction log.
- `learning.correction_examples()` turns a log into explicit hard examples: a merge is a boundary the model
  predicted and a human rejected (hard negative); a split is one the model missed (hard positive).
- For the 9 existing labels, the same information is recorded in `ground_truth/provenance.json`.
- Saving a label never trains anything. Training is a separate, explicit command.

## Dataset roles (`ground_truth/splits.json`)

| role | used for | may be trained/tuned on? |
|---|---|---|
| `dev` | training and development (leave-one-video-out inside) | yes |
| `val` | held-out validation: accept or reject a model | **no** |
| `test` | final test, evaluated once after the model is frozen | **never** |

A label that is not listed in any role is ignored by `evaluate`, `tune` and `train` until someone assigns it, so
nothing can leak by accident. `train` refuses to run if a training video is also listed under `val` or `test`,
and it never loads `test` videos at all.

Prefer labels made with **Start blank** for `val` and `test`. Labels made by editing predictions are anchored to
the current rule and favour it in any comparison.

## Training

```bash
python -m sceneseen train            # ~2 min; add --no-tuned to skip the tuned-rule comparison
```

1. Loads `dev` (training) and `val` (validation). `test` is never loaded.
2. Builds the cut-level examples.
3. Compares, on videos each method did not see:
   A current rule · B grid-tuned rule · C logistic regression · C gradient boosting (80 depth-2 trees) ·
   D hybrid (the rule's boundaries, vetoed or extended by the classifier when it is confident).
   Hyperparameters and thresholds are chosen by leave-one-video-out **inside the training videos only**.
4. Applies the acceptance rule: better than the current rule on dev leave-one-video-out **and** not worse on
   validation **and** better on more videos than worse.
5. Saves `models/boundary_logreg_<hash>.json` and `reports/learning_<timestamp>.{md,json}`.

The saved artifact is plain JSON (no pickle; loading needs only numpy). It records the training videos with the
SHA-256 of each label file, a ground-truth version hash, the features, algorithm, hyperparameters, random seed,
dev and validation metrics, the current rule's metrics, the accept/reject decision and an artifact hash. The hash
excludes the creation time, so retraining on identical data gives the identical version. A file edited after
training is refused on load.

## Using a model (only if accepted)

```toml
[boundary_model]
path = "models/boundary_logreg_<hash>.json"
```

Default is `path = ""`: the hand-designed rule. If the configured file is missing, edited or built for a different
feature set, SceneSeen logs a warning and falls back to the rule.

## Result of the first training run (2026-10-03)

Macro F1@±2s:

| method | training fit (in-sample) | dev, leave-one-video-out | held-out validation |
|---|---|---|---|
| **A current rule** | 0.677 | **0.677** | **0.812** |
| B tuned rule | 0.735 | 0.606 | 0.694 |
| C logistic regression | 0.716 | 0.580 | 0.706 |
| C gradient boosting | 0.770 | 0.698 | 0.694 |
| D hybrid | 0.677 | 0.671 | 0.812 |

**All rejected. The current rule stays the default.** Every trained variant looks better in-sample and worse on
unseen videos, which is overfitting with 53 positive examples. Gradient boosting edges ahead on dev (0.698) but
falls to 0.694 against 0.812 on validation. The hybrid chose to change nothing.

Caveat that keeps the question open: 6 of the 9 labels were made by editing the rule's own predictions, which
favours the rule. On the 3 labels made from blank, the picture reverses (rule 0.53, logistic 0.59, boosting 0.63),
but 3 videos is far too few to act on. More **from-blank** labels are what would settle it. Per-video numbers are
in docs/RESULTS.md and `reports/learning_*.md`.
