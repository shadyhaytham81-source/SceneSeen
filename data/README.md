# data/ — local only, never committed

Everything in this folder except this README is ignored by Git. Videos are large and copyrighted; caches and
exports are regenerated automatically.

```
data/
├── videos/     ← put the labelled videos here (YOU create this and copy the files in)
├── uploads/    ← files uploaded through the web app (created automatically)
├── cache/      ← shots, CLIP features, thumbnails per video (created automatically, safe to delete)
├── exports/    ← scenes.json + scene clips (created automatically)
└── synthetic/  ← optional sanity-check videos from scripts/make_synthetic.py
```

## The labelled dataset

Each file in `ground_truth/*.json` names its video in the `"video"` field. Copy that video into
`data/videos/` with **exactly that file name**, including odd ones like `video_01.mp4.mp4`. Arabic and emoji names
are fine: matching ignores Unicode normalisation differences between macOS, Windows and Linux.

`ground_truth/videos_manifest.json` lists the exact size and SHA-256 of every labelled video. A different
download (another quality or re-encode) can shift timestamps, so results would no longer be comparable. Check that
everything is in place:

```bash
python -m sceneseen check-data            # presence, readability, size
python -m sceneseen check-data --verify   # + SHA-256 (reads every file, ~10 s per GB)
```

Every line should show `[x] … ok`. Then reproduce the evaluation:

```bash
python -m sceneseen evaluate --split dev
```

It should print SceneSeen F1@2s = 0.677 (see docs/RESULTS.md). The first run takes a while (about 1–2 min of
shot detection per 10 min of video). After that everything is cached in `data/cache/`.
