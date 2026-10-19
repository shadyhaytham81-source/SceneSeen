"""Scene-level context: what kind of place is this scene? (no bounding boxes)

"Restaurant", "office" or "indoor" are properties of a whole scene, so they are classified,
not detected. This reuses the OpenCLIP frame embeddings Phase 1 already cached: each context
in the taxonomy is a few text prompts, embedded once with the same CLIP model (and cached on
disk), and compared with every shot by cosine similarity. No video is decoded and, once the
text embeddings exist, no model is loaded.

Zero-shot CLIP is right often, not always, so the answer is only reported when it is both
confident and clearly ahead of the runner-up; otherwise the venue is "unknown". The taxonomy
contains non-commercial settings (street, hospital, vehicle ...) so the classifier is not forced
to pick a commercial venue.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ..shots import Shot
from .taxonomy import ENVIRONMENTS, VENUES, Context

log = logging.getLogger(__name__)
LOGIT_SCALE = 100.0
MIN_EVIDENCE = 0.10   # at least this share of a scene must show the place (not close-ups / titles)


def _key(model: str, pretrained: str, facets: dict[str, dict[str, Context]]) -> str:
    src = json.dumps([model, pretrained, {f: {c.id: c.prompts for c in d.values()} for f, d in facets.items()}])
    return hashlib.sha1(src.encode()).hexdigest()[:12]


def text_embeddings(cache_root: Path, model: str, pretrained: str, device: str = "cpu") -> dict[str, np.ndarray]:
    """{prompt: unit embedding} for every context prompt, computed once and cached on disk."""
    facets = {"venue": VENUES, "environment": ENVIRONMENTS}
    path = cache_root / "_models" / f"context_text_{_key(model, pretrained, facets)}.npz"
    prompts = [p for d in facets.values() for c in d.values() for p in c.prompts]
    if path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            if list(data["prompts"]) == prompts:
                return dict(zip(prompts, data["emb"]))
        except Exception as e:  # corrupted cache file: recompute
            log.warning("ignoring unreadable context cache %s (%s)", path.name, e)
    import open_clip
    import torch

    from ..features import load_clip

    clip, _ = load_clip(model, pretrained, device)
    tok = open_clip.get_tokenizer(model)
    with torch.no_grad():
        emb = clip.encode_text(tok(prompts).to(device)).float()
    emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prompts=np.array(prompts), emb=emb)
    return dict(zip(prompts, emb))


def _facet_probs(shot_emb: np.ndarray, contexts: dict[str, Context], text: dict[str, np.ndarray]) -> np.ndarray:
    """[n_shots, n_contexts] softmax over contexts (best prompt per context)."""
    cols = []
    for c in contexts.values():
        T = np.stack([text[p] for p in c.prompts])
        cols.append((shot_emb @ T.T).max(axis=1))
    logits = LOGIT_SCALE * np.stack(cols, axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def classify_scenes(clip: np.ndarray, shots: list[Shot], scenes: list[dict], text: dict[str, np.ndarray],
                    min_confidence: float, min_margin: float) -> list[dict]:
    """Per scene: venue (or None when unsure), environment, and the ranked alternatives."""
    e = clip.astype(np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-9)
    shot_emb = e.mean(axis=1)
    shot_emb /= np.maximum(np.linalg.norm(shot_emb, axis=-1, keepdims=True), 1e-9)
    pv, pe = _facet_probs(shot_emb, VENUES, text), _facet_probs(shot_emb, ENVIRONMENTS, text)
    dur = np.array([max(s.end - s.start, 1e-3) for s in shots])
    keys = list(VENUES)
    shown = np.array([VENUES[k].show for k in keys])
    # A close-up or a title card says nothing about the place. Each shot votes with a weight of
    # (its duration) x (how much of its probability is on real venues), and its vote is spread
    # over the real venues only.
    informative = pv[:, shown].sum(axis=1)
    pv_shown = np.where(shown[None, :], pv, 0.0) / np.maximum(informative[:, None], 1e-9)
    out = []
    for sc in scenes:
        if sc.get("shot_start") is None:
            out.append({"venue": None, "environment": None, "ranked": []})
            continue
        ids = np.arange(sc["shot_start"], sc["shot_end"] + 1)
        w_env = dur[ids] / dur[ids].sum()
        env = (w_env[:, None] * pe[ids]).sum(axis=0)
        wv = dur[ids] * informative[ids]
        evidence = float(wv.sum() / dur[ids].sum())          # share of the scene that shows the place at all
        v = (wv[:, None] * pv_shown[ids]).sum(axis=0) / max(wv.sum(), 1e-9)
        order = np.argsort(-v)
        ranked = [{"id": keys[i], "label": VENUES[keys[i]].label, "confidence": round(float(v[i]), 3)} for i in order[:3]]
        top = VENUES[keys[order[0]]]
        sure = (evidence >= MIN_EVIDENCE and v[order[0]] >= min_confidence
                and (v[order[0]] - v[order[1]]) >= min_margin)
        venue = {"id": top.id, "label": top.label, "icon": top.icon, "category": top.category,
                 "confidence": round(float(v[order[0]]), 3), "evidence": round(evidence, 3)} if sure else None
        ek = list(ENVIRONMENTS)
        ei = int(np.argmax(env))
        environment = {"id": ek[ei], "label": ENVIRONMENTS[ek[ei]].label, "confidence": round(float(env[ei]), 3)} \
            if env[ei] >= 0.6 else None
        out.append({"venue": venue, "environment": environment, "ranked": ranked})
    return out
