"""Configuration loading: default.toml + optional override files, as a nested dict
with typed accessors for the sections each stage needs."""
from __future__ import annotations

import os
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
class CatalogConfig:
    """Phase 2B: brand / product catalogue storage."""
    database_url: str = "sqlite:///data/catalog/catalog.db"   # PostgreSQL in production: postgresql+psycopg://...
    images_dir: str = "data/catalog/images"


@dataclass(frozen=True)
class MatchingConfig:
    """Phase 2B: product matching. Thresholds come from scripts/product_matching_study.py."""
    embedder: str = "openclip_b32"
    device: str = "auto"
    batch_size: int = 32
    colour_weight: float = 0.5           # score = cosine + colour_weight * colour-histogram intersection
    max_occurrences: int = 5             # occurrence crops of one object used as evidence
    crop_pad: float = 0.08
    shortlist: int = 200
    top_k: int = 5
    # match confidence = sigmoid(conf_w_score * score + conf_w_margin * margin + conf_bias), where margin is
    # the lead over the runner-up product (the signal that separates a real match from a lookalike).
    conf_w_score: float = 2.703
    conf_w_margin: float = 18.441
    conf_bias: float = -4.991
    high_confidence: float = 0.80        # held-out: top candidate correct 93 % of the time
    possible_confidence: float = 0.50    # 58 %
    uncertain_confidence: float = 0.25   # 50 %; below: no reliable match (9 %)
    background_score: float = 1.047      # typical best score of a WRONG product (75th percentile in the study)
    min_products_for_high: int = 5       # fewer comparable products: runner-up floored, state capped at "possible"
    calibration_version: str = "2026-10-05/openclip_b32/colour0.5/62-objects"


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
    catalog: CatalogConfig = CatalogConfig()
    matching: MatchingConfig = MatchingConfig()

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
        catalog=_build(CatalogConfig, raw.get("catalog", {})),
        matching=_build(MatchingConfig, raw.get("matching", {})),
    )


def catalog_paths(cfg: "Config") -> tuple[str, Path]:
    """(database URL, image folder) with relative locations resolved against the project root."""
    url = os.environ.get("SCENESEEN_DATABASE_URL") or cfg.catalog.database_url
    if url.startswith("sqlite:///") and not Path(url[len("sqlite:///"):]).is_absolute():
        url = "sqlite:///" + str(ROOT / url[len("sqlite:///"):])
    img = Path(cfg.catalog.images_dir)
    return url, (img if img.is_absolute() else ROOT / img)


def config_hash(section) -> str:
    """Stable short hash of a config section, used to key cache entries."""
    blob = json.dumps(asdict(section), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
