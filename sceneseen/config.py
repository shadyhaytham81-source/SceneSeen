"""Configuration loading: default.toml + optional override files, as a nested dict
with typed accessors for the sections each stage needs."""
from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass, asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class ShotsConfig:
    detector: str = "transnetv2"
    transnet_threshold: float = 0.5
    transnet_device: str = "cpu"
    pyscenedetect_threshold: float = 3.0
    min_shot_seconds: float = 0.4


@dataclass(frozen=True)
class FeaturesConfig:
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    batch_size: int = 64
    frame_positions: tuple = (0.15, 0.5, 0.85)
    frame_short_side: int = 224
    color_bins: tuple = (8, 4, 4)


@dataclass(frozen=True)
class GroupingConfig:
    clip_weight: float = 0.75
    color_weight: float = 0.25
    clip_floor: float = 0.5
    window_shots: int = 6
    window_seconds: float = 45.0
    distance_penalty: float = 0.01
    coherence_topk: int = 2
    abs_threshold: float = 0.45
    depth_threshold: float = 0.35
    strong_threshold: float = 0.75
    depth_mode: str = "climb"
    min_scene_seconds: float = 6.0

    def replace(self, **kw) -> "GroupingConfig":
        d = asdict(self)
        unknown = set(kw) - set(d)
        if unknown:
            raise ValueError(f"unknown grouping params: {sorted(unknown)}")
        d.update(kw)
        return GroupingConfig(**d)


@dataclass(frozen=True)
class UniqueShotsConfig:
    """Repeated-shot grouping (post-processing; no effect on scene segmentation).
    Weights/thresholds were calibrated on labelled shot pairs: see docs/UNIQUE_SHOTS.md."""
    w_clip: float = 2.776
    w_color: float = 3.540
    w_phash: float = 4.089
    bias: float = -7.886
    duplicate_threshold: float = 0.60
    different_threshold: float = 0.35
    phash_neutral: float = 0.60
    linkage: str = "average"
    rep_w_centrality: float = 0.35
    rep_w_sharpness: float = 0.25
    rep_w_duration: float = 0.20
    rep_w_stability: float = 0.10
    rep_w_exposure: float = 0.10


@dataclass(frozen=True)
class CommercialConfig:
    """Phase 2A: commercial objects + scene context (runs on top of Phase 1; never changes it)."""
    detector: str = "owlv2"              # owlv2 | grounding_dino  (commercial/detector.py)
    device: str = "auto"                 # auto -> cuda / mps / cpu
    batch_size: int = 4
    frame_long_side: int = 1280          # representative frames are extracted at this size
    frame_positions: tuple = (0.5,)      # relative positions inside each unique shot's representative
    store_threshold: float = 0.10        # detections kept in the cache (so display thresholds can change freely)
    min_confidence: float = 0.40         # global floor; each object type adds its own tier (taxonomy.py)
    min_relevance: float = 0.45          # ... and this commercial relevance
    describe_colors: bool = True
    context_min_confidence: float = 0.60   # scene venue is reported only when this sure ...
    context_min_margin: float = 0.15       # ... and this far ahead of the runner-up; otherwise "unknown"


@dataclass(frozen=True)
class CatalogueConfig:
    """Phase 2B: product catalogues for visual matching (dataset lives OUTSIDE Git, see docs/MATCHING.md)."""
    # Folders that contain one sub-folder per brand (<brand>/catalogue.jsonl + images/ + embeddings).
    # Relative paths are resolved against the project root. The environment variable
    # SCENESEEN_CATALOGUE_DIR (several folders separated by ':' on macOS/Linux, ';' on Windows)
    # replaces this list, so each teammate can point at their own copy of the dataset.
    dirs: tuple = ("scraper/output", "data/catalogues")
    model: str = "ViT-B-32"              # embedding model, same for catalogue images and object crops
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"
    batch_size: int = 64
    top_k: int = 3
    crop_pad: float = 0.08               # padding around the detected box (fraction of box size)
    color_same: float = 0.04             # ranking adjustment added to cosine similarity
    color_near: float = 0.02
    color_different: float = -0.04
    color_different_multicolor: float = -0.01


@dataclass(frozen=True)
class BoundaryModelConfig:
    """Optional learned scene-boundary classifier. Empty path = hand-designed rule (default)."""
    path: str = ""


@dataclass(frozen=True)
class ExportConfig:
    clip_mode: str = "reencode"
    crf: int = 18
    preset: str = "veryfast"


@dataclass(frozen=True)
class Paths:
    cache_dir: Path
    uploads_dir: Path
    exports_dir: Path
    ground_truth_dir: Path
    videos_dir: Path


@dataclass(frozen=True)
class Config:
    shots: ShotsConfig
    features: FeaturesConfig
    grouping: GroupingConfig
    export: ExportConfig
    paths: Paths
    unique_shots: UniqueShotsConfig = UniqueShotsConfig()
    boundary_model: BoundaryModelConfig = BoundaryModelConfig()
    commercial: CommercialConfig = CommercialConfig()
    catalogue: CatalogueConfig = CatalogueConfig()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, section: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(section) - names
    if unknown:
        raise ValueError(f"unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in section.items()}
    return cls(**kw)


def load_config(*override_files: str | Path | None) -> Config:
    raw = tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for f in override_files:
        if f:
            raw = _merge(raw, tomllib.loads(Path(f).read_text(encoding="utf-8")))
    paths = {k: (ROOT / v if not Path(v).is_absolute() else Path(v)) for k, v in raw["paths"].items()}
    return Config(
        shots=_build(ShotsConfig, raw.get("shots", {})),
        features=_build(FeaturesConfig, raw.get("features", {})),
        grouping=_build(GroupingConfig, raw.get("grouping", {})),
        export=_build(ExportConfig, raw.get("export", {})),
        paths=Paths(**paths),
        unique_shots=_build(UniqueShotsConfig, raw.get("unique_shots", {})),
        boundary_model=_build(BoundaryModelConfig, raw.get("boundary_model", {})),
        commercial=_build(CommercialConfig, raw.get("commercial", {})),
        catalogue=_build(CatalogueConfig, raw.get("catalogue", {})),
    )


def catalogue_dirs(cfg: "CatalogueConfig | None" = None) -> list[Path]:
    """Folders searched for brand catalogues: $SCENESEEN_CATALOGUE_DIR if set, else [catalogue] dirs."""
    import os

    env = os.environ.get("SCENESEEN_CATALOGUE_DIR", "").strip()
    raw = [x for x in env.split(os.pathsep) if x.strip()] if env else list((cfg or CatalogueConfig()).dirs)
    out = []
    for d in raw:
        p = Path(d).expanduser()
        out.append(p if p.is_absolute() else ROOT / p)
    return out


def config_hash(section) -> str:
    """Stable short hash of a config section, used to key cache entries."""
    blob = json.dumps(asdict(section), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
