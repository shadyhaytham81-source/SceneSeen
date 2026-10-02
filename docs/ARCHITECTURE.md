# SceneSeen Phase 1 — Architecture

## Pipeline

```
               ┌──────────── cached per video: data/cache/<video_id>/ ─────────────┐
video ─► probe ─► shot detection ─► frame sampling ─► CLIP + colour ─► grouping ─► scenes
         media.py  shots.py          features.py       features.py       grouping.py   export.py
                   shots_<cfg>.json  thumbs_<cfg>/     features_<cfg>.npz  (ms, not cached)
```

* **video_id** is a content hash (size + first/last 4 MB), so a renamed or re-uploaded file reuses its cache.
* Every cache entry is keyed by a hash of the config section that produced it. Changing `[shots]` invalidates
  shots and features. Changing `[features]` invalidates features only. Changing `[grouping]` invalidates
  nothing, because grouping is re-run from cached features in milliseconds.
* Modules do not depend on the web layer. `server/` only calls `pipeline`, `export`, `corrections` and
  `ground_truth`.

## Decisions and why

### 1. Shot detection: TransNetV2 by default
TransNetV2 is a small 3D-CNN trained for shot boundary detection, and its weights ship inside the pip package. It
detects gradual transitions (dissolves, fades), and those are disproportionately *scene* transitions. On the
Sintel trailer it found 14 shots where PySceneDetect's AdaptiveDetector found 9.
- Runs on CPU. MPS gave identical predictions in testing but was slower.
- Speed is about 0.15× real time (12-min film: 112 s), and the model is compute-bound. Batching windows and adding
  threads gave no gain. `detector = "pyscenedetect"` is about 10× faster for hard-cut-only content.
- Post-processing (`predictions_to_cuts`, `merge_short_shots`) is deterministic and unit-tested. A transition
  spanning several frames becomes one cut at its midpoint. Shots shorter than 0.4 s (flashes) are absorbed.

### 2. Shot representation: 3 frames, mean embedding
Three frames at 15 %, 50 % and 85 % of each shot cover pans and reveals without decoding every frame. A single
ffmpeg decode at 224 px short side feeds a streaming batcher, so memory is flat for long videos. Per frame:
- **OpenCLIP ViT-B/32 (LAION-2B)**: semantic embedding (place, setting, people, time of day). It is small and fast
  (12-min film ≈ 15 s on MPS), and its semantics fit "same place / same situation" better than pixel features.
- **HSV histogram (8×4×4)**: lighting and palette continuity. It is cheap, and it separates day/night or
  warm/cold versions of similar content.

Known limitation: CLIP centre-crops to a square, so wide 2.39:1 frames lose their sides.

### 3. Scene grouping: windowed coherence (the core)
Classic but explainable: Kender & Yeo 1998 ("continuous video coherence") and Hanjalic et al. 1999
("logical story units"), with CLIP as the shot descriptor.

1. Shot similarity `S = 0.75·CLIP' + 0.25·colour`, where CLIP' rescales cosine similarity from [0.5, 1] to [0, 1].
2. For every cut *k*, compare every shot in a window **before** the cut with every shot in a window **after** it
   (6 shots and at most 45 s apart). Coherence = mean of the top-2 links, minus 0.01 per intermediate shot.
   **This is what handles A→B→A→B**: when A returns after the cut, it matches the A before the cut, so the cut
   gets high coherence and is not a boundary. Near video edges, k shrinks to the number of shots available.
3. Boundary score = 1 − coherence. A cut becomes a scene boundary if:
   - score ≥ `abs_threshold` (0.45), **and**
   - its **TextTiling depth** (Hearst 1997: rise above the nearest valleys on both sides) ≥ `depth_threshold`
     (0.35), **or** score ≥ `strong_threshold` (0.75);
   - then greedy non-maximum suppression keeps scenes ≥ `min_scene_seconds` (6 s).

Why depth instead of a global z-score: the first version used a per-video robust z-score. On *Tears of Steel*
it merged a blue night exterior into a warm church interior (cut score 0.63), because in a scene-dense film
boundaries are not statistical outliers. Depth is local, so it adapts to each film's editing rhythm. The strong
threshold covers the opposite failure: a run of consecutive unrelated scenes forms a plateau with zero depth.

Every cut's score, depth, best link (which shots matched) and decision are kept for the developer view.

### 4. What was deliberately not done
- **No model training.** Supervised scene models (e.g. BaSSL, LGSS on MovieNet-SSeg) need large labelled
  corpora and give opaque decisions. Phase 1 uses pretrained features and an explainable grouping layer. A learned
  boundary classifier on top of the same features is the natural next step once labels exist.
- **No audio** yet. It is easy to add as another similarity term (ambience/music continuity). It will only be
  added if evaluation shows it helps, because adding signals without measurement is guesswork.
- **No object detection.** It is not needed for segmentation (Phase 2 concern).

### 5. Exports
- JSON: scene metadata, with manual corrections applied when they exist.
- Clips: **re-encode (libx264, CRF 18) by default.** This was verified on a real cut: exact frame count, and the
  first frame matches the scene's first source frame (PSNR 43.7 dB vs 10.9 dB against the previous frame).
  Stream copy (`clip_mode = "copy"`) is lossless and instant, but in testing it was off by frames even when the
  scene started on a keyframe, so it is opt-in.
- Browsers can't play every container/codec. When the source isn't H.264/HEVC/VP9/AV1 in MP4/MOV/WEBM, a 480p
  preview proxy is generated once and cached.

### 6. Corrections
Split (snapped to the nearest real shot cut within 1 s) and merge (with the next scene). They are stored in
`data/cache/<id>/corrections.json` as the corrected boundary list plus an append-only log. Each merge is a
labelled false boundary and each split a labelled missed boundary, which is future training signal for a learned
boundary classifier.

## Phase 2 readiness
Phase 2 (objects, clothing, places per scene) plugs in without a rewrite:
- Scene records already carry `shot_start`/`shot_end`, and shots carry frame ranges. A Phase 2 module can iterate
  scenes → shots → sampled frames, and `features.sample_frame_indices` is reusable.
- Per-shot CLIP embeddings are already cached. They double as zero-shot place/setting classifiers (text prompts)
  and as a retrieval index.
- `pipeline.load_stage_outputs` is the integration point: a Phase 2 stage adds its own cached artifact keyed the
  same way, without touching shots or grouping.
- The web layer is a thin API over the package, and new endpoints/panels are additive.

## Scaling notes
For 45–60 min episodes: decoding is streamed, TransNetV2 holds 48×27 frames (≈ 350 MB for 90k frames), and
grouping is O(shots × window²). The single-worker job runner is enough for a local tool. A queue (e.g. RQ) is the
upgrade path if this ever becomes multi-user.
