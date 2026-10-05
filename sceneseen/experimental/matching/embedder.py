"""Image embedders for product matching, behind one interface.

    embedder.embed(list of RGB arrays) -> float32 [N, D], L2-normalised

Pretrained models only. Each embedder has a `key` that names the model AND its weights
revision; embeddings are cached under that key, so changing the model (or its version) never
mixes old and new vectors, and an unchanged image is never embedded twice with the same model.
Loaded lazily, once per process. Any failure raises EmbedderUnavailable (callers fall back to
showing generic commercial objects without product matches).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


class EmbedderUnavailable(RuntimeError):
    """The embedding model cannot be used (download failed, out of memory, missing library)."""


@dataclass(frozen=True)
class EmbedderSpec:
    name: str
    backend: str          # open_clip | hf
    model_id: str
    pretrained: str = ""
    license: str = ""
    pooling: str = "cls"  # hf models: cls | cls_mean | image_features


EMBEDDERS: dict[str, EmbedderSpec] = {s.name: s for s in [
    EmbedderSpec("openclip_b32", "open_clip", "ViT-B-32", "laion2b_s34b_b79k", "MIT"),
    EmbedderSpec("dinov2_small", "hf", "facebook/dinov2-small", license="Apache-2.0", pooling="cls"),
    EmbedderSpec("dinov2_base", "hf", "facebook/dinov2-base", license="Apache-2.0", pooling="cls"),
    EmbedderSpec("siglip_base", "hf", "google/siglip-base-patch16-224", license="Apache-2.0", pooling="image_features"),
    EmbedderSpec("marqo_ecommerce_b", "open_clip", "hf-hub:Marqo/marqo-ecommerce-embeddings-B", license="Apache-2.0"),
]}


def _device(preferred: str) -> str:
    import torch

    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hub_revision(model_id: str) -> str:
    """Commit hash of the cached Hugging Face snapshot (part of the cache key)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(model_id, "config.json")
        if isinstance(p, str) and "/snapshots/" in p.replace("\\", "/"):
            return p.replace("\\", "/").split("/snapshots/")[1].split("/")[0][:12]
    except Exception:
        pass
    return "unknown"


class Embedder:
    def __init__(self, spec: EmbedderSpec, device: str = "auto", batch_size: int = 32):
        self.spec, self.device_pref, self.batch_size = spec, device, batch_size
        self.device: str | None = None
        self.key = self.cache_key()
        self.dim: int | None = None
        self.load_seconds: float | None = None
        self._model = self._pre = None
        self._lock = threading.Lock()

    def cache_key(self) -> str:
        """The key embeddings are cached under, WITHOUT loading the model (used to answer from
        cache). Identical to `key` after loading."""
        if self.spec.backend == "open_clip":
            return f"{self.spec.name}:{self.spec.model_id}:{self.spec.pretrained or 'hub'}"
        return f"{self.spec.name}:{self.spec.model_id}:{_hub_revision(self.spec.model_id)}"

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            t = time.perf_counter()
            try:
                if self.spec.backend == "open_clip":
                    import open_clip

                    if self.spec.pretrained:
                        model, _, pre = open_clip.create_model_and_transforms(self.spec.model_id,
                                                                              pretrained=self.spec.pretrained)
                    else:
                        model, _, pre = open_clip.create_model_and_transforms(self.spec.model_id)
                    revision = self.spec.pretrained or "hub"
                else:
                    import transformers as T

                    pre = T.AutoImageProcessor.from_pretrained(self.spec.model_id)
                    model = T.AutoModel.from_pretrained(self.spec.model_id)
                    revision = _hub_revision(self.spec.model_id)
            except Exception as e:
                raise EmbedderUnavailable(f"could not load {self.spec.model_id}: {type(e).__name__}: {e}") from e
            model.eval()
            self.device = _device(self.device_pref)
            try:
                model.to(self.device)
            except Exception as e:
                log.warning("moving the embedder to %s failed (%s); using CPU", self.device, e)
                self.device = "cpu"
                model.to("cpu")
            self._model, self._pre = model, pre
            self.key = f"{self.spec.name}:{self.spec.model_id}:{revision}"
            self.load_seconds = round(time.perf_counter() - t, 2)
            log.info("embedder %s ready on %s in %.1fs", self.key, self.device, self.load_seconds)

    def _forward(self, images: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        pil = [Image.fromarray(im) for im in images]
        with torch.no_grad():
            if self.spec.backend == "open_clip":
                x = torch.stack([self._pre(p) for p in pil]).to(self.device)
                out = self._model.encode_image(x).float()
            else:
                inp = self._pre(images=pil, return_tensors="pt").to(self.device)
                if self.spec.pooling == "image_features":
                    out = self._model.get_image_features(**inp)
                    out = getattr(out, "pooler_output", out)
                else:
                    h = self._model(**inp).last_hidden_state
                    out = h[:, 0] if self.spec.pooling == "cls" else torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
                out = out.float()
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return out.cpu().numpy().astype(np.float32)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        self.load()
        if not images:
            return np.zeros((0, self.dim or 0), np.float32)
        parts = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i:i + self.batch_size]
            try:
                parts.append(self._forward(chunk))
            except Exception as e:
                if self.device == "cpu":
                    raise EmbedderUnavailable(f"embedding failed: {type(e).__name__}: {e}") from e
                log.warning("embedder failed on %s (%s); retrying on CPU", self.device, e)
                self.device = "cpu"
                self._model.to("cpu")
                try:
                    parts.append(self._forward(chunk))
                except Exception as e2:
                    raise EmbedderUnavailable(f"embedding failed on CPU: {type(e2).__name__}: {e2}") from e2
        out = np.concatenate(parts)
        self.dim = out.shape[1]
        return out

    def describe(self) -> dict:
        return {"key": self.key, "name": self.spec.name, "model": self.spec.model_id, "license": self.spec.license,
                "device": self.device, "dim": self.dim, "load_seconds": self.load_seconds}


_EMBEDDERS: dict[tuple, Embedder] = {}
_LOCK = threading.Lock()


def get_embedder(name: str, device: str = "auto", batch_size: int = 32) -> Embedder:
    """Process-wide embedder instance (the model is loaded once and reused)."""
    if name not in EMBEDDERS:
        raise EmbedderUnavailable(f"unknown embedder '{name}' (available: {', '.join(EMBEDDERS)})")
    k = (name, device, batch_size)
    with _LOCK:
        if k not in _EMBEDDERS:
            _EMBEDDERS[k] = Embedder(EMBEDDERS[name], device, batch_size)
        return _EMBEDDERS[k]
