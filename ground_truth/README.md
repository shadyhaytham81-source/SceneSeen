# Ground truth labels

Human-made scene boundary labels, one JSON per video. The matching video goes in `data/videos/`.

```json
{"video": "video_01.mp4", "boundaries": [48.2, 127.4, 221.8], "annotator": "shady", "notes": ""}
```

`splits.json` gives every label a role: `dev` (training/development), `val` (held-out validation) or `test`
(final test). **A new label has no role and is ignored until you add its file name (without `.json`) to one of
those lists.** `provenance.json` records how each label was made, `videos_manifest.json` the exact video files,
and `unique_shots/` the shot-pair labels used to calibrate Unique Shots. Labelling rules and the UI workflow are in
[../docs/EVALUATION.md](../docs/EVALUATION.md); how labels are used for training is in
[../docs/LEARNING.md](../docs/LEARNING.md).

Never put model predictions or synthetic data here without a human checking them.
