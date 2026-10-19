# SceneSeen — Context & Architecture Guide

## Overview & Purpose
**SceneSeen** is a video intelligence system that segments video into **narrative scenes** rather than individual camera cuts (e.g. dialogue cuts `A → B → A → B` are kept as one scene).
- **Phase 1 (Complete):** Local shot detection, semantic/color feature extraction, windowed-coherence scene grouping, and within-scene repeated camera setup detection (**Unique Shots**).
- **Phase 2A (Complete):** Commercial scene understanding — detects 38 commercial object types (Fashion, Electronics, Automotive, Home, etc.) using open-vocabulary detection (OWLv2) on unique shot representatives, plus scene-level venue/environment context using OpenCLIP zero-shot classification.
- **Phase 2B+ (Future / Not started):** Product catalog matching, exact brand/product identification, placement opportunities, viewer shop-the-episode features.

Everything runs **100% locally** (no external cloud APIs required; static FFmpeg bundled via `imageio-ffmpeg`).

---

## High-Level Architecture & Pipeline

```
Video Input (Local file or web upload)
  │
  ├── 1. Media Probe (sceneseen/media.py)
  │      - Content-based hash ID (size + head/tail 4MB) for instant cache reuse
  │      - Extraction of duration, fps, resolution, SAR/DAR
  │
  ├── 2. Shot Detection (sceneseen/shots.py)
  │      - TransNetV2 (3D-CNN, CPU-based) catches hard cuts and gradual transitions (fades/dissolves)
  │      - PySceneDetect available as a fast alternative for hard-cut content
  │      - Deterministic post-processing: merges cuts < 0.4s
  │      - Output cached: data/cache/<video_id>/shots_<cfg>.json
  │
  ├── 3. Feature Extraction (sceneseen/features.py)
  │      - Samples 3 frames per shot (at 15%, 50%, 85% of duration)
  │      - OpenCLIP ViT-B/32 (LAION-2B): semantic embeddings (place, setting, people)
  │      - HSV Color Histogram (8×4×4): lighting and color palette continuity
  │      - Output cached: data/cache/<video_id>/features_<cfg>.npz + thumbnails
  │
  ├── 4. Scene Grouping (sceneseen/grouping.py)
  │      - Computes shot similarity: S = 0.75 · CLIP' + 0.25 · HSV
  │      - Windowed Coherence: compares shots before vs after cut (up to 6 shots / 45s)
  │      - Boundary Score = 1 - Coherence
  │      - Boundary decision: (score ≥ 0.45 AND TextTiling depth ≥ 0.35) OR (score ≥ 0.75)
  │      - Non-maximum suppression enforcing minimum scene length (≥ 6.0s)
  │      - Runs in milliseconds from cached features (no training required)
  │
  ├── 5. Unique Shots Post-Processing (sceneseen/unique.py & similarity.py)
  │      - Groups repeated camera setups within each scene (averages ~41% shot reduction)
  │      - Probability: p(dup) = sigmoid(2.776·clip_matched + 3.540·color + 4.089·phash - 7.886)
  │      - Grouped if p ≥ 0.60; picks best representative frame by sharpness/exposure/duration
  │
  ├── 6. Phase 2A: Commercial Scene Understanding (sceneseen/commercial/)
  │      - Detector: OWLv2-base on 1 representative frame per unique shot
  │      - Scene Context: Zero-shot classification with CLIP embeddings (venue & environment)
  │      - Deduplication & Commercial Relevance scoring
  │      - Output cached: data/cache/<video_id>/commercial/
  │
  └── 7. Web UI & Export (server/app.py & sceneseen/export.py)
         - Frame-accurate video clip re-encoding (libx264, CRF 18)
         - Interactive timeline, video playback, scene editor (split/merge), developer mode
```

---

## Key Files & Directory Structure

### Core Pipeline (`sceneseen/`)
- `pipeline.py`: Orchestrator connecting probe, shot detection, feature extraction, grouping, and caching.
- `media.py`: Video probing, content-hash generation, bundled FFmpeg stream decoding.
- `shots.py`: TransNetV2 & PySceneDetect integration, transition probability peak finding, short shot merging.
- `features.py`: Frame sampling, OpenCLIP batch inference, HSV histogram extraction, thumbnail saving.
- `grouping.py`: Windowed cross-cut coherence calculation, TextTiling depth analysis, NMS boundary detection.
- `similarity.py`: Vectorized shot similarity metrics (multi-frame CLIP cosine, HSV, pHash, SSIM).
- `unique.py`: Within-scene camera setup clustering, representative shot selection.
- `export.py`: JSON generation and frame-accurate clip extraction via FFmpeg.
- `corrections.py`: Interactive scene split/merge persistence and provenance tracking.
- `config.py`: TOML configuration loader and validator with hierarchical overrides.
- `__main__.py`: Unified CLI entry point (`analyze`, `serve`, `commercial`, `evaluate`, `tune`, `train`, `check-data`).

### Phase 2A Commercial Understanding (`sceneseen/commercial/`)
- `detector.py`: Hugging Face OWLv2 open-vocabulary object detector wrapper.
- `taxonomy.py`: 38 object types across 7 commercial categories + detector prompts + confidence tiers.
- `relevance.py`: Commercial relevance scoring (base type value × size, centrality, persistence).
- `scene_context.py`: CLIP zero-shot classification for venue (restaurant, office, etc.) and indoor/outdoor.
- `dedup.py`: In-frame, in-shot, and across-scene candidate deduplication.
- `analysis.py`: End-to-end commercial pipeline orchestration and caching.
- `review.py`: Human/AI review tracking (Correct / Wrong / Missed) and precision reporting.

### Evaluation & Machine Learning (`sceneseen/`)
- `ground_truth.py`: Ground truth schema, JSON validation, boundary loading, dataset split management.
- `evaluation.py`: Precision, Recall, F1@±1/2/3s, timing error, coverage & overflow metrics.
- `benchmark.py`: Dataset loader, baseline comparisons (`all_shots`, `adjacent`), LOVO threshold grid-tuning.
- `learning.py`: Shot-cut feature extraction (16 features) and boundary classifier training (disabled by default; rule-based remains superior).

### Configuration & Data Directories
- `config/default.toml`: Canonical parameters for shots, features, grouping, export, unique shots, and commercial detection.
- `server/app.py`: FastAPI server serving REST API and static frontend assets (`server/static/`).
- `ground_truth/`: Human scene boundary annotations, `splits.json` (`dev`, `val`, `test`), `provenance.json`, and `videos_manifest.json`.
- `data/`: Local working storage (git-ignored): `videos/` (ground truth videos), `uploads/` (web uploads), `cache/` (intermediate features), `exports/` (rendered scenes and clips).
- `reports/`: Markdown and JSON logs of evaluation runs and threshold tuning.
- `scripts/`: Diagnostic and benchmark utilities (`make_synthetic.py`, `error_analysis.py`, `unique_shots_study.py`, `commercial_benchmark.py`).

---

## Ground Truth & Evaluation Rules
1. **Roles in `ground_truth/splits.json`:**
   - `dev` (6 videos): Training and development (leave-one-video-out tuning).
   - `val` (3 videos): Unseen validation to accept/reject models.
   - `test` (0 videos currently): Final frozen test set evaluated once.
2. **Anchoring Caveat:**
   - Labels made by editing predictions (merging/splitting SceneSeen output) score optimistically (F1 ~0.87).
   - Labels created using **Start Blank** give the least-biased performance metric (real films F1 ~0.60).
3. **Current Benchmark Scores (Default Rule):**
   - Dev Set: **F1@±2s = 0.677** (vs 0.14 for every cut, 0.32 for adjacent similarity).
   - Validation Set: **F1@±2s = 0.812**.
   - Learned classifiers (logistic regression, gradient boosting) were evaluated and rejected because they overfit on small sample counts.

---

## Quick Testing Commands Summary
- **Fast Test Suite:** `pytest -k "not slow"`
- **Start Web Application:** `python -m sceneseen serve` (open http://127.0.0.1:8000)
- **CLI Video Analysis:** `python -m sceneseen analyze <video_path> [--clips] [--debug]`
- **Commercial Analysis:** `python -m sceneseen commercial <video_path> [--all]`
- **Check Dataset:** `python -m sceneseen check-data [--verify]`
