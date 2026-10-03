# SceneSeen — Phase 1: Video Scene Segmentation

SceneSeen takes a video and returns its **narrative scenes**, not just its camera cuts.

```
video → shot detection → shot features → scene grouping → scenes (timestamps, previews, JSON, clips)
```

A dialogue cut as `A → B → A → B` is four shots but one scene. SceneSeen keeps it as one scene and starts a new
scene when the setting, light or people change. Everything runs locally: no paid APIs, no cloud.

Project status, results and next steps: **[PROJECT_STATUS.md](PROJECT_STATUS.md)**.

## Setup (new machine)

**Requirements:** Python **3.11** (tested; 3.12 should work), Git, about 3 GB free disk (dependencies ≈ 1.5 GB,
CLIP model ≈ 0.6 GB, plus cache). Internet access is needed once, for `pip install` and the first CLIP download.

| Dependency | How you get it |
|---|---|
| FFmpeg | **Nothing to install.** `imageio-ffmpeg` ships a static ffmpeg binary inside the venv (Mac/Windows/Linux). A system ffmpeg is not used. |
| TransNetV2 weights (shot detection) | Bundled inside the `transnetv2-pytorch` pip package. |
| OpenCLIP ViT-B/32 (LAION-2B) weights | Downloaded automatically from Hugging Face on the first analysis (≈ 600 MB, cached in `~/.cache/huggingface`). No account or token needed. |
| GPU | Optional. Uses Apple MPS or CUDA for CLIP if present, otherwise CPU. Shot detection always runs on CPU. |

### macOS

```bash
# 0. One-time: Python 3.11 (skip if `python3.11 --version` already works)
brew install python@3.11

# 1. Clone
git clone https://github.com/shadyhaytham81-source/SceneSeen.git
# 2. Go into the project
cd SceneSeen
# 3. Create a virtual environment
python3.11 -m venv .venv
# 4. Activate it (do this in every new terminal)
source .venv/bin/activate
# 5. Install dependencies (~1.5 GB, includes PyTorch and a bundled FFmpeg)
python -m pip install --upgrade pip
pip install -r requirements.txt
# 6. FFmpeg / models: nothing to do. FFmpeg is bundled; the CLIP model downloads itself on first analysis.
# 7. Run the tests (fast suite, a few seconds)
python -m pytest
# 8. Start SceneSeen
python -m sceneseen serve
# 9. Open http://127.0.0.1:8000 in your browser (Ctrl+C in the terminal stops the server)
```

### Windows 10/11 (PowerShell)

```powershell
# 0. One-time: install Python 3.11 from https://www.python.org/downloads/ (tick "Add python.exe to PATH")
#    and Git from https://git-scm.com/download/win, then check:
py -3.11 --version

# 1. Clone
git clone https://github.com/shadyhaytham81-source/SceneSeen.git
# 2. Go into the project
cd SceneSeen
# 3. Create a virtual environment
py -3.11 -m venv .venv
# 4. Activate it (do this in every new terminal). If PowerShell blocks scripts, run once:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
# 5. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
# 6. FFmpeg / models: nothing to do (bundled FFmpeg; CLIP downloads itself on first analysis)
# 7. Run the tests
python -m pytest
# 8. Start SceneSeen
python -m sceneseen serve
# 9. Open http://127.0.0.1:8000 in your browser
```

(Command Prompt instead of PowerShell: activate with `.venv\Scripts\activate.bat`.)

> **macOS + iCloud:** do not keep the project in a folder synced by iCloud Drive with "Optimise Mac Storage"
> (Desktop/Documents). macOS evicts videos, caches and virtualenv files to the cloud and reads then hang. Clone
> somewhere like `~/Projects/`, or at least name the environment `.venv.nosync` (iCloud ignores `*.nosync`).

The first video you analyse downloads the CLIP model (~600 MB, once). Shot detection takes about 1–2 minutes per
10 minutes of video on a laptop CPU. Results are cached, so re-opening a video is instant.

## Videos are NOT stored in GitHub

Videos are large and copyrighted. Git only contains code, docs, labels and evaluation results. There are two ways
videos enter SceneSeen:

**1. New videos: just upload them in the website.** Open http://127.0.0.1:8000 and drag the file in (or **Choose
Video**). The app keeps its own copy in `data/uploads/`. You do **not** need to put anything in `data/videos/`
yourself. When you save a ground-truth label, SceneSeen copies the video into `data/videos/` automatically.

**2. The existing labelled dataset (6 videos), to reproduce the evaluation.** Get the files from the project owner
(shared separately), copy them into **`data/videos/`** with their exact file names, then verify:

```bash
python -m sceneseen check-data --verify   # each label must show [x] ok, sha256 match
python -m sceneseen evaluate --split dev  # reproduces docs/RESULTS.md (SceneSeen F1@±2s = 0.677)
```

`check-data` compares each file with `ground_truth/videos_manifest.json` (exact size and SHA-256), so you know
you have the identical files. Without the videos it lists them as `MISSING` and exits; nothing hangs. Details:
[data/README.md](data/README.md).

## Labelling a video (creating ground truth)

1. **Upload** the video on the home page and click **Analyze Video**.
2. Tick **Developer** (top right).
3. Turn on **Edit scenes**, then click **Start blank**. This removes SceneSeen's prediction, so it can't
   influence you.
4. **Watch the entire video.** Each time a new scene really starts, pause and click **Split at playhead**. It
   snaps to the real cut just before where you paused.
5. **Review** the scene cards and timeline. Fix mistakes with **Merge with next** or another split.
6. Click **Mark as reviewed**.
7. Click **Save as ground truth** and enter your name as annotator. This writes `ground_truth/<video>.json` and
   copies the video to `data/videos/`.
8. Give the label a role: add its name to `dev`, `val` or `test` in `ground_truth/splits.json`. Labels without a
   role are ignored. Labels made with Start blank are the ones to use for `val` and `test`.
9. Commit the new label (`git add ground_truth/ && git commit`). **Never commit the video.**

> **Rule: a camera cut or a different camera angle does NOT automatically mean a new scene.** A conversation
> filmed as close-up A → close-up B → wide shot is **one** scene. Start a new scene only when the **location**,
> the **time** (next day, flashback), or the story's **situation** changes. Full rules and edge cases:
> [docs/EVALUATION.md](docs/EVALUATION.md).

### Optional: end-to-end tests on real video

Uses the open movie *Tears of Steel* (Blender Foundation, CC-BY, 372 MB):

```bash
mkdir -p data/videos
curl -L -o data/videos/tears_of_steel_720p.mov https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
python scripts/make_synthetic.py
python -m pytest -m slow
```

On Windows PowerShell use `curl.exe` (not `curl`) and `mkdir data\videos`.

## Using SceneSeen

**Web app:** after analysis you get scene cards and a timeline. **Preview** plays one scene. **Export Scene
Data** downloads JSON, and **Export Scene Clips** downloads a zip of `scene_001.mp4`, …. **Developer** mode shows
shots, cut scores, grouping decisions, thresholds, representative frames and timings, and can re-run grouping
with new parameters instantly (preview only).

**Unique Shots:** below the scene cards, every scene lists Original / Unique / Repeated shots and the reduction.
Repeated camera set-ups are grouped under one representative thumbnail; click it to see every original occurrence,
or switch to **All Shots**. Nothing is removed. Details: [docs/UNIQUE_SHOTS.md](docs/UNIQUE_SHOTS.md).

**Command line**

```bash
python -m sceneseen analyze path/to/video.mp4            # prints scenes, writes data/exports/<name>/scenes.json
python -m sceneseen analyze path/to/video.mp4 --clips    # + one frame-accurate clip per scene
python -m sceneseen analyze path/to/video.mp4 --debug    # + debug.json (shots, scores, decisions)
python -m sceneseen check-data [--verify]                # are all labelled videos in data/videos/ (and identical)?
python -m sceneseen evaluate --split dev                 # score against human labels + baselines
python -m sceneseen tune                                 # tune thresholds on dev only (with leave-one-out check)
python -m sceneseen evaluate --split val                 # held-out validation videos
python -m sceneseen train                                # train + validate the optional boundary classifier
python scripts/error_analysis.py                         # diagnose every false/missed boundary
python scripts/unique_shots_study.py evaluate            # re-run the Unique Shots similarity comparison
```

## Output

```json
{
  "video": "movie_clip.mp4",
  "duration": 374.2,
  "scene_count": 5,
  "scenes": [
    {"scene_id": 1, "start_seconds": 0.0, "end_seconds": 48.2, "duration_seconds": 48.2},
    {"scene_id": 2, "start_seconds": 48.2, "end_seconds": 127.4, "duration_seconds": 79.2}
  ]
}
```

## How it works (short)

| Stage | Choice | Why |
|---|---|---|
| Shot detection | **TransNetV2** (CPU), PySceneDetect as a fast option | Neural detector; also catches dissolves and fades, which often mark scene changes |
| Shot features | 3 frames per shot (15/50/85 %) → **OpenCLIP ViT-B/32** embedding + HSV colour histogram | CLIP captures setting/people/time-of-day; colour captures lighting continuity |
| Grouping | **Windowed coherence** + TextTiling depth + minimum scene length | Links recurring set-ups across a window of shots, so A/B dialogue stays together; no training needed |
| Export | JSON; clips re-encoded with libx264 | Verified frame-accurate. Stream copy is an option but only keyframe-accurate |

Details and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Repeated-shot grouping:
[docs/UNIQUE_SHOTS.md](docs/UNIQUE_SHOTS.md). What is trainable and how ground truth is used:
[docs/LEARNING.md](docs/LEARNING.md). Evaluation protocol and labelling
guide: [docs/EVALUATION.md](docs/EVALUATION.md).

## Measured performance (Apple Silicon laptop, CPU + MPS)

| Video | Length | Shots | Scenes | Shot detection | Frames + CLIP | Grouping | 23 clips export |
|---|---|---|---|---|---|---|---|
| Tears of Steel 720p | 12:14 | 140 | 23 | 112 s | 15 s | 4 ms | 36 s |
| Synthetic test (854×356) | 3:14 | 51 | 6 | 30 s | 6 s | 3 ms | — |

Shot detection dominates, at roughly 0.15× the video duration. Set `detector = "pyscenedetect"` in config for a
roughly 10× faster but less robust detector. Everything expensive is cached per video, so re-grouping with new
thresholds takes milliseconds.

## Status

Phase 1 is built, tested (automated test suite: `python -m pytest`) and evaluated on 6 labelled videos: **F1@2s 0.68**, vs 0.14 for "every cut is
a scene" and 0.32 for adjacent-shot similarity. See [PROJECT_STATUS.md](PROJECT_STATUS.md) and
[docs/RESULTS.md](docs/RESULTS.md). **Phase 2 has not started and must not start yet.**

## Project layout

```
sceneseen/            core package (no web code)
  media.py            probe, content-hash video IDs, frame decoding (bundled ffmpeg)
  shots.py            shot boundary detection + deterministic post-processing
  features.py         frame sampling, CLIP embeddings, colour histograms, thumbnails
  grouping.py         shot → scene grouping (pure numpy) + baselines
  pipeline.py         orchestration + cache
  export.py           JSON + clip export
  corrections.py      split / merge
  ground_truth.py     label format, validation, frozen dev/test split
  evaluation.py       metrics (P/R/F1 @ tolerance, timing error, coverage/overflow)
  benchmark.py        evaluation runner, baselines, threshold tuning with leave-one-out
  similarity.py       vectorised shot-to-shot similarity measures (CLIP cosine, colour, pHash, SSIM)
  unique.py           Unique Shots: repeated-shot grouping per scene (post-processing)
  learning.py         boundary examples from ground truth, optional classifier, versioned artifacts
server/               FastAPI app + static UI (plain HTML/CSS/JS)
config/default.toml   all thresholds and model choices
ground_truth/         human scene-boundary labels (9 videos), dev/val/test roles, provenance, video manifest
models/               versioned trained boundary-model artifacts (JSON; none enabled by default)
data/                 LOCAL ONLY: videos, uploads, caches, exports (see data/README.md)
reports/              evaluation reports (JSON/markdown) and error analysis (images not committed)
docs/                 ARCHITECTURE.md, EVALUATION.md (protocol + labelling guide), RESULTS.md
scripts/              synthetic sanity-video builder, error analysis
tests/                unit + end-to-end tests
```
