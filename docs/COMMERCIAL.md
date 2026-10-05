# Phase 2A — Commercial Scene Understanding

Phase 2A sits on top of Phase 1 and never changes it:

```
Video → Scenes → Unique Shots ─► 1 representative frame per unique shot ─► open-vocabulary detector
          (Phase 1, unchanged)                                              │
                                              cached CLIP embeddings ─► scene context (venue, indoor/outdoor)
                                                                            ▼
                         taxonomy → relevance → de-duplication → commercial objects per scene
```

It answers "which commercially relevant things are in this scene, and where do they appear?". It does **not**
identify brands or products ("sneakers", never "Nike Air Force 1"). That is Phase 2B (product catalogue matching).

## Two levels of understanding

| | Object level | Scene level |
|---|---|---|
| Examples | smartphone, watch, sofa, car | home interior, office, retail store, indoor/outdoor |
| How | open-vocabulary **detector** on representative frames | zero-shot **classification** of Phase 1's cached CLIP embeddings |
| Has a bounding box | yes | no, it is a property of the whole scene |
| Module | `commercial/detector.py` | `commercial/scene_context.py` |

Scene-level concepts are never forced into object boxes.

## Model selection (measured on this project's footage)

42 representative frames (1280×720) from 6 SceneSeen videos, 30 queries, Apple M4 / 16 GB, PyTorch MPS.
Raw numbers: `reports/commercial/model_study.json`.

| model | licence | s / frame (MPS) | accelerator memory | verdict |
|---|---|---|---|---|
| **OWLv2 base (ensemble)** | Apache-2.0 | 0.64 | 1.15 GB | **Selected** |
| Grounding DINO tiny | Apache-2.0 | 4.59 | 3.48 GB | Rejected |
| Florence-2 base | MIT | 0.59 | 1.34 GB | Rejected |
| OmDet-Turbo (Swin-T) | Apache-2.0 | 0.17 | 1.34 GB | Rejected |

- **OWLv2** takes any number of text queries and returns a calibrated score **per query**, with clean labels. That
  is what makes a precision threshold per object type possible. It runs on MPS, and on CPU at 1.46 s/frame.
- **Grounding DINO** was 7× slower on Apple Silicon (several operations fall back to CPU) and used 3× the memory.
  With many categories in one prompt it returns merged phrases ("sneakers shoes", "smartphone tablet") that
  cannot be mapped to a taxonomy.
- **Florence-2**: 167 of 253 detections were "man / woman / human face"; it found 3 phones and no watches, bags or
  jewelry. It returns **no confidence scores**, so precision cannot be controlled. Its open-vocabulary mode needs
  one generation per query.
- **OmDet-Turbo** is by far the fastest, but about half of a blind sample of 48 detections at ≥ 0.4 were wrong
  (a channel logo as "lamp", a face as "television"). It is worth revisiting only if speed becomes the bottleneck.
- Not evaluated: MM-Grounding-DINO (same family and prompt format as Grounding DINO); YOLO-World / YOLOE (AGPL-3.0,
  unsuitable for a commercial product); Grounding DINO 1.5 / DINO-X (API only).
- **SAM 2 is not used.** Masks and tracking add nothing to "which commercial objects are in this scene"; boxes are
  enough for this phase.

Nothing is trained. OWLv2's weights (~620 MB) download from Hugging Face on first use.

## Performance design

- **Unique Shots decide what gets analysed.** A camera set-up that occurs ten times costs one inference; its
  detections are credited to all ten occurrences. On the 9 labelled videos this skips 41 % of shots (up to 64 %).
- **One frame per unique shot** (the middle frame). Measured against three frames (15/50/85 %) on two videos:
  3× the inference time for 7 extra objects per video, all in the borderline 0.41–0.66 confidence band. Not worth
  it; `frame_positions` in the config switches it on.
- The model loads **once per server process**, lazily, on the first analysis. Reading cached results never loads it.
- Batched inference (4 frames). 85 % of the time is OWLv2's vision backbone.

| measurement (Apple M4, MPS) | value |
|---|---|
| Model load (weights already downloaded) | 4–5 s |
| Inference per unique shot, 48 queries | 0.85–0.95 s (CPU: 1.46 s) |
| Representative frame extraction | ~0.05 s per frame (4 in parallel) |
| 2:22 clip (Dark Knight, 25 unique shots), end to end incl. model load | 23 s |
| 12:02 episode clip (117 shots → 45 unique) | 42 s |
| 9:48 film opening (156 shots → 124 unique) | ~2 min |
| Same videos again (cached) | 0.04–0.3 s, no model loaded |
| Memory | ~1 GB process + ~2.2 GB accelerator at batch 4; 2.3 GB on CPU |

Benchmark on any machine: `python scripts/commercial_benchmark.py --device mps|cpu`.

## Taxonomy (`commercial/taxonomy.py`)

38 object types in 7 categories (Fashion, Accessories, Electronics, Automotive, Food & Beverage, Furniture & Home,
Beauty & Personal Care), plus scene contexts mapped to Locations & Venues (restaurant, café, hotel, gym, retail
store, bar) and Real Estate & Environment (home interior, office, villa, residential compound). Each object type
has detector prompts, a commercial base value and a confidence tier. Generic things (person, wall, door, window)
are simply not in the taxonomy, so they are never asked for.

## Commercial relevance (`commercial/relevance.py`), deterministic V1

```
relevance = base value of the type × (0.60 + 0.20·size + 0.08·centrality + 0.12·persistence)
```

`base` is how likely a brand would pay for, or a viewer would shop for, that kind of thing (watch 0.95, chair 0.50).
Presentation only scales it between 60 % and 100 %: a prominent chair never outranks a visible watch, but a tiny
background chair falls below the display threshold (0.45). Hidden candidates stay visible in Developer mode with
the reason.

## De-duplication (`commercial/dedup.py`)

1. **Inside a frame:** overlapping boxes for the same thing are merged: same type, competing types of one group
   (sneakers vs shoes, sunglasses vs eyeglasses, sofa vs armchair vs chair), or any two labels on the same box.
2. **Inside a shot** (only when several frames are sampled): the same object seen twice is kept once.
3. **Across a scene:** all detections of one object type become **one candidate** with its list of shot
   occurrences, e.g. "Smartphone · seen ×7 · 05:15–06:55". Occurrences include repeats of the analysed camera
   set-up (marked as not individually analysed).

## Measured precision

Reviews were done **blind by the AI assistant** from crops with the proposed label and no scores shown. They are
**not human-verified**. They are stored in `ground_truth/commercial_reviews/` in the same format the UI writes, so
a human can confirm or overrule each one (Developer mode → ✓ Correct / ✗ Wrong / Missed).

**Step 1: calibration (4 videos, 177 judged detections, sampled evenly across confidence):**

| detector confidence | precision |
|---|---|
| 0.20–0.30 | 15 % |
| 0.30–0.40 | 39 % |
| 0.40–0.50 | 75 % |
| 0.50–0.60 | 75 % |
| ≥ 0.60 | 94 % |

Precision also depends on the kind of object: large distinctive things (trousers 6/6, suit 6/7, car 3/3) are right
early; small or rare things were almost never right (camera 0/6, headphones 0/5, tablet 0/2, "lipstick" firing on
lips 0/4). Hence four display tiers instead of one threshold:

| tier | min confidence | types |
|---|---|---|
| large, distinctive | 0.40 | clothing, hats, cars, motorcycles, sofas, chairs, tables, beds |
| default | 0.50 | shoes, bags, eyewear, phones, laptops, lamps, bottles, glasses |
| small accessories | 0.55 | watch, jewelry |
| rarely right | 0.65 | tablet, headphones, television, camera, soda can, coffee cup, food packaging, appliances, cosmetics, perfume, skincare |

**Step 2: held-out check (5 videos NOT used for calibration; 160 of the 164 displayed objects, sampled uniformly):**

| | correct | wrong | unsure | precision |
|---|---|---|---|---|
| **Displayed commercial objects** | 118 | 21 | 21 | **84.9 %** (unsure excluded); 73.8 % if every unsure counts as wrong |

Per video: 87 %, 84 %, 83 %, 92 %, 80 %. By type: suit 14/14, jacket 7/7, sofa 7/7, shirt 5/5, armchair 5/5,
television 4/4, handbag 4/4, lamp 11/12, table 9/10, eyeglasses 9/10, car 7/8, jewelry 6/7, bed 5/6, watch 5/7,
smartphone 5/7, chair 8/12, sunglasses 1/3.

Recall was not measured (precision is the priority). The "Missed" control records objects a reviewer notices.

```bash
python -m sceneseen commercial-report                      # precision of all reviews under the current rules
python -m sceneseen commercial-report --min-confidence 0.6 # same reviews, different threshold
```

**Scene context** (venue): 11 of 16 reported venues were right on held-out videos (≈ 70 %). Home interior 6/6,
office and street market were right; "restaurant" fired on a family dinner at home and on an office meeting. A
venue is only reported when confidence ≥ 0.60 and 0.15 ahead of the runner-up, and close-ups / title cards barely
vote; otherwise it is "unknown" (about a third of scenes).

## Caching

Everything is under `data/cache/<video_id>/commercial/`, separate from Phase 1's files:

| file | key | invalidated by |
|---|---|---|
| `frames_<size>/shot_XXXX_pYYY.jpg` | shot id + position | frame size; new unique-shot representatives add frames |
| `detections_<key>.json` | **frame content hash** → raw detections | model, model settings, **detector prompts** |

Relevance values, display thresholds, labels, icons, de-duplication and scene context are applied afterwards from
the cached raw detections. Changing them costs milliseconds and never re-runs the model. Nothing here can
invalidate Phase 1. If Phase 1 changes (scenes corrected, different unique shots), only representative frames
that were never analysed need inference. No absolute paths are stored. A corrupted cache file is set aside and
rebuilt.

## API

| endpoint | purpose |
|---|---|
| `GET /api/videos/{id}/commercial` | cached results; never loads the model. `status`: `ready`, `partial`, `not_run`, `unavailable` |
| `POST /api/videos/{id}/commercial/analyze` | start the analysis job (progress via `/api/jobs/commercial-{id}`) |
| `GET /api/videos/{id}/commercial/frame/{shot}/{pos}?box=…&size=…` | representative frame or padded crop |
| `POST /api/videos/{id}/commercial/review` | Correct / Wrong for a candidate, or a missed object |
| `GET /api/commercial/report` | precision of reviewed detections |

Each candidate carries what Phase 2B (catalogue matching) and Phase 3 (placement opportunities) need: type,
category, confidence, relevance, colour attribute, the best frame + box (crop source), and every occurrence with
timestamps and unique-shot ids. Neither phase is implemented.

CLI: `python -m sceneseen commercial VIDEO [--all]` writes `commercial.json` next to `scenes.json`.

## Failure behaviour

Commercial analysis can fail without affecting scenes or Unique Shots:
- model download or load fails → `status: "unavailable"` with the reason; the UI shows
  "Commercial analysis unavailable" and a retry button;
- MPS / CUDA error → the detector moves to CPU and retries;
- video file missing (e.g. evicted to iCloud) → frames already extracted are still used (`partial`);
- unreadable frame → skipped and counted; corrupted cache → rebuilt.

## Known failure cases

- **Sunglasses vs eyeglasses** and **chair vs anything chair-shaped** are the most frequent wrong labels.
- **Small objects in dark or blurred frames** (watch, phone) are missed or wrong; one mid-shot frame can be blurred.
- **Garment types** are approximate: a hoodie may be "jacket", a T-shirt "shirt", a waistcoat "suit".
- **Colours** are only added when one colour clearly dominates, and coloured lighting can still mislead them.
- **Rare types** (camera, headphones, tablet, cosmetics, perfume, food packaging) are mostly hidden by their 0.65
  tier, so real ones are often missed. This is deliberate until there is evidence they can be trusted.
- **Occurrences are inherited from the camera set-up.** A phone seen in the analysed frame is credited to every
  repeat of that set-up, even if the actor put it down in between.
- **Scene context** is the weakest part (≈ 70 %): "restaurant" for any group at a table, wrong venues for dark
  rooms. Treat it as a hint.
- **On-screen graphics** (channel logos, subtitles, tickers) are not removed before detection.
- Logos and text on products are not read; nothing identifies a brand.
