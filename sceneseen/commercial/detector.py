"""Open-vocabulary object detectors behind one small interface.

    detector.detect(images, prompts) -> one list of (prompt_index, score, box) per image

* Pretrained models from the `transformers` library; nothing is trained here.
* Loaded lazily, ONCE per process (thread-safe), and reused for every request.
* Device: Apple MPS or CUDA when available, otherwise CPU. If the accelerator fails (unsupported
  operation, out of memory) the detector moves to CPU and retries instead of failing the job.
* Any failure to obtain or run the model raises DetectorUnavailable: callers degrade to
  "commercial analysis unavailable" and Phase 1 keeps working.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # single unsupported ops fall back to CPU


class DetectorUnavailable(RuntimeError):
    """The commercial detector cannot be used (download failed, out of memory, missing library)."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str            # owlv2 | grounding_dino
    license: str
    prompt_template: str   # how a taxonomy prompt becomes a text query
    native_threshold: float  # scores below this are noise for this model family


MODELS = {
    "owlv2": ModelSpec("owlv2", "google/owlv2-base-patch16-ensemble", "owlv2", "Apache-2.0", "a photo of a {}", 0.10),
    "grounding_dino": ModelSpec("grounding_dino", "IDEA-Research/grounding-dino-tiny", "grounding_dino", "Apache-2.0",
                                "{}", 0.20),
}


def pick_device(preferred: str = "auto") -> str:
    import torch

    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFDetector:
    """OWLv2 / Grounding DINO via transformers."""

    def __init__(self, spec: ModelSpec, device: str = "auto", store_threshold: float = 0.10, batch_size: int = 4):
        self.spec = spec
        self.name = spec.model_id
        self.device_pref = device
        self.device: str | None = None
        self.store_threshold = max(store_threshold, spec.native_threshold)
        self.batch_size = batch_size
        self.load_seconds: float | None = None
        self.version = ""
        self._model = self._proc = None
        self._lock = threading.Lock()

    # ---- loading
    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            t = time.perf_counter()
            try:
                import transformers as T

                self._proc = T.AutoProcessor.from_pretrained(self.spec.model_id)
                model = T.AutoModelForZeroShotObjectDetection.from_pretrained(self.spec.model_id)
                self.version = f"transformers {T.__version__}; {getattr(model.config, '_commit_hash', None) or 'local'}"
            except Exception as e:  # network, disk, missing package, corrupt download ...
                raise DetectorUnavailable(f"could not load {self.spec.model_id}: {type(e).__name__}: {e}") from e
            model.eval()
            self.device = pick_device(self.device_pref)
            try:
                model.to(self.device)
            except Exception as e:
                log.warning("moving the detector to %s failed (%s); using CPU", self.device, e)
                self.device = "cpu"
                model.to("cpu")
            self._model = model
            self.load_seconds = round(time.perf_counter() - t, 2)
            log.info("commercial detector %s ready on %s in %.1fs", self.spec.model_id, self.device, self.load_seconds)

    def _to_cpu(self, reason: Exception) -> None:
        log.warning("detector failed on %s (%s: %s); retrying on CPU", self.device, type(reason).__name__, reason)
        self.device = "cpu"
        self._model.to("cpu")

    # ---- inference
    def _run(self, images: list, queries: list[str]) -> list[list[tuple[int, float, list[float]]]]:
        import torch

        sizes = [(im.height, im.width) for im in images]
        with torch.no_grad():
            if self.spec.family == "owlv2":
                inp = self._proc(text=[queries] * len(images), images=images, return_tensors="pt").to(self.device)
                out = self._model(**inp)
                res = self._proc.post_process_grounded_object_detection(
                    out, threshold=self.store_threshold, target_sizes=sizes, text_labels=[queries] * len(images))
            else:
                inp = self._proc(images=images, text=[queries] * len(images), return_tensors="pt").to(self.device)
                out = self._model(**inp)
                res = self._proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=self.store_threshold, text_threshold=self.store_threshold,
                    target_sizes=sizes)
        results = []
        index = {q: i for i, q in enumerate(queries)}
        for r, (h, w) in zip(res, sizes):
            labels = r.get("text_labels") or r.get("labels")
            dets = []
            for lab, score, box in zip(labels, r["scores"].tolist(), r["boxes"].tolist()):
                qi = index.get(lab) if isinstance(lab, str) else int(lab)
                if qi is None:      # Grounding DINO can return merged phrases; keep only exact queries
                    continue
                x1, y1, x2, y2 = box
                nb = [min(max(x1 / w, 0.0), 1.0), min(max(y1 / h, 0.0), 1.0),
                      min(max(x2 / w, 0.0), 1.0), min(max(y2 / h, 0.0), 1.0)]
                if nb[2] - nb[0] > 0.004 and nb[3] - nb[1] > 0.004:
                    dets.append((qi, float(score), nb))
            results.append(dets)
        return results

    def detect(self, images: list[np.ndarray], prompts: list[str]) -> list[list[tuple[int, float, list[float]]]]:
        """Per image: [(prompt_index, score, [x1, y1, x2, y2] as fractions of the frame)]."""
        from PIL import Image

        self.load()
        queries = [self.spec.prompt_template.format(p) for p in prompts]
        pil = [Image.fromarray(im) for im in images]
        out: list = []
        for i in range(0, len(pil), self.batch_size):
            chunk = pil[i:i + self.batch_size]
            try:
                out += self._run(chunk, queries)
            except Exception as e:
                if self.device == "cpu":
                    raise DetectorUnavailable(f"detector inference failed: {type(e).__name__}: {e}") from e
                self._to_cpu(e)
                try:
                    out += self._run(chunk, queries)
                except Exception as e2:
                    raise DetectorUnavailable(f"detector inference failed on CPU: {type(e2).__name__}: {e2}") from e2
        return out

    def describe(self) -> dict:
        return {"name": self.spec.model_id, "family": self.spec.family, "license": self.spec.license,
                "version": self.version, "device": self.device, "load_seconds": self.load_seconds}


_DETECTORS: dict[tuple, HFDetector] = {}
_REG_LOCK = threading.Lock()


def get_detector(key: str, device: str = "auto", store_threshold: float = 0.10, batch_size: int = 4) -> HFDetector:
    """Process-wide detector instance (the model is loaded once and reused)."""
    if key not in MODELS:
        raise DetectorUnavailable(f"unknown detector '{key}' (available: {', '.join(MODELS)})")
    k = (key, device, store_threshold, batch_size)
    with _REG_LOCK:
        if k not in _DETECTORS:
            _DETECTORS[k] = HFDetector(MODELS[key], device, store_threshold, batch_size)
        return _DETECTORS[k]
