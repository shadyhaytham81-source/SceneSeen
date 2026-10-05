"""Benchmark the Phase 2A detector on this machine.

    python scripts/commercial_benchmark.py [--device mps|cpu|auto] [--frames 12]

Uses representative frames already cached by a previous commercial analysis (run
`python -m sceneseen commercial VIDEO` or the web app first). Reports model load time,
seconds per frame (first call and steady state), and memory.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sceneseen.commercial import taxonomy as T  # noqa: E402
from sceneseen.commercial.detector import HFDetector, MODELS  # noqa: E402
from sceneseen.commercial.frames import load_rgb  # noqa: E402
from sceneseen.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    c = cfg.commercial
    files = sorted(cfg.paths.cache_dir.glob(f"*/commercial/frames_{c.frame_long_side}/shot_*_p050.jpg"))[:: 7][: a.frames]
    if not files:
        sys.exit("no cached frames found; run a commercial analysis first")
    images = [load_rgb(f) for f in files]
    prompts = [p for p, _ in T.detector_queries()]
    det = HFDetector(MODELS[a.detector or c.detector], a.device, c.store_threshold, a.batch or c.batch_size)
    t = time.perf_counter()
    det.load()
    load = time.perf_counter() - t
    t = time.perf_counter()
    det.detect(images[:1], prompts)
    first = time.perf_counter() - t
    t = time.perf_counter()
    res = det.detect(images, prompts)
    steady = (time.perf_counter() - t) / len(images)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    gpu = None
    try:
        import torch

        if det.device == "mps":
            gpu = torch.mps.driver_allocated_memory() / 1e9
        elif det.device == "cuda":
            gpu = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    print(json.dumps({"model": det.name, "device": det.device, "batch_size": det.batch_size, "frames": len(images),
                      "frame_size": list(images[0].shape[1::-1]), "prompts": len(prompts),
                      "model_load_seconds": round(load, 2), "first_frame_seconds": round(first, 2),
                      "seconds_per_frame": round(steady, 3), "peak_process_memory_gb": round(rss, 2),
                      "accelerator_memory_gb": None if gpu is None else round(gpu, 2),
                      "detections_at_store_threshold": sum(len(r) for r in res)}))


if __name__ == "__main__":
    main()
