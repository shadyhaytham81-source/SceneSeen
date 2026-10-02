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
# 1. Python 3.11 (skip if `python3.11 --version` works)
brew install python@3.11

# 2. Clone
git clone <REPO_URL> sceneseen
cd sceneseen

# 3. Virtual environment + dependencies
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Tests (fast unit tests, ~3 s)
python -m pytest

# 5. Start SceneSeen -> open http://127.0.0.1:8000
python -m sceneseen serve
```

### Windows 10/11 (PowerShell)

```powershell
# 1. Python 3.11 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then check:
py -3.11 --version

# 2. Clone
git clone <REPO_URL> sceneseen
cd sceneseen

# 3. Virtual environment + dependencies
py -3.11 -m venv .venv
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Tests
python -m pytest

# 5. Start SceneSeen -> open http://127.0.0.1:8000
python -m sceneseen serve
```

(Command Prompt instead of PowerShell: activate with `.venv\Scripts\activate.bat`.)

### Dataset (videos are NOT in Git)

The ground-truth labels are in `ground_truth/`. The matching videos must be copied separately into
**`data/videos/`**, each with exactly the file name given in its label's `"video"` field. See
[data/README.md](data/README.md). Then:

```bash
python -m sceneseen check-data          # every label should show [x] ok
python -m sceneseen evaluate --split dev # reproduces docs/RESULTS.md (SceneSeen F1@2s = 0.677)
```

### Optional: end-to-end tests on real video

Uses the open movie *Tears of Steel* (Blender Foundation, CC-BY, 372 MB):

```bash
curl -L -o data/videos/tears_of_steel_720p.mov https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov
python scripts/make_synthetic.py
python -m pytest -m slow
```

On Windows PowerShell use `curl.exe` (not `curl`), and create the folder first with `mkdir data\videos`.

## Using SceneSeen

**Web app** (`python -m sceneseen serve`): drop a video, click **Analyze Video**, then preview scenes and export
**Scene Data** (JSON) or **Scene Clips** (zip of `scene_001.mp4`, …). Tick **Developer** (top right) to see shots,
cut scores, grouping decisions, thresholds, representative frames and timings, and to re-run grouping with new
parameters instantly. **Edit scenes** lets you merge and split scenes. In developer mode you can save the result
as a ground-truth label (see docs/EVALUATION.md, "Start blank" for unbiased labels).

**Command line**

```bash
python -m sceneseen analyze path/to/video.mp4            # prints scenes, writes data/exports/<name>/scenes.json
python -m sceneseen analyze path/to/video.mp4 --clips    # + one frame-accurate clip per scene
python -m sceneseen analyze path/to/video.mp4 --debug    # + debug.json (shots, scores, decisions)
python -m sceneseen check-data                           # are all labelled videos in data/videos/?
python -m sceneseen evaluate --split dev                 # score against human labels + baselines
python -m sceneseen tune                                 # tune thresholds on dev only (with leave-one-out check)
python scripts/error_analysis.py                         # diagnose every false/missed boundary
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

Details and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Evaluation protocol and labelling
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

Phase 1 is built, tested (41 tests) and evaluated on 6 labelled videos: **F1@2s 0.68**, vs 0.14 for "every cut is
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
server/               FastAPI app + static UI (plain HTML/CSS/JS)
config/default.toml   all thresholds and model choices
ground_truth/         human scene-boundary labels (6 videos) + frozen dev/test split
data/                 LOCAL ONLY: videos, uploads, caches, exports (see data/README.md)
reports/              evaluation reports (JSON/markdown) and error analysis (images not committed)
docs/                 ARCHITECTURE.md, EVALUATION.md (protocol + labelling guide), RESULTS.md
scripts/              synthetic sanity-video builder, error analysis
tests/                unit + end-to-end tests
```
