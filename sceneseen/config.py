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
    )


def config_hash(section) -> str:
    """Stable short hash of a config section, used to key cache entries."""
    blob = json.dumps(asdict(section), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
