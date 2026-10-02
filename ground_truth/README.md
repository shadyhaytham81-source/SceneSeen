# Ground truth labels

Human-made scene boundary labels, one JSON per video. The matching video goes in `data/videos/`.

```json
{"video": "video_01.mp4", "boundaries": [48.2, 127.4, 221.8], "annotator": "shady", "notes": ""}
```

`splits.json` (dev/test) is created automatically the first time you run `python -m sceneseen evaluate` or
`tune`, and is frozen after that. Labelling rules and the UI workflow are in
[../docs/EVALUATION.md](../docs/EVALUATION.md).

Never put model predictions or synthetic data here without a human checking them.
