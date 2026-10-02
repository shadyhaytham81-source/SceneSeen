"""End-to-end checks on real decoded video (slow; skipped when the media is absent).

    python scripts/make_synthetic.py
    python -m pytest -m slow
"""
import json

import pytest

from sceneseen import grouping
from sceneseen.config import ROOT, load_config
from sceneseen.evaluation import evaluate
from sceneseen.export import export_clips
from sceneseen.media import iter_frames, probe
from sceneseen.pipeline import analyze, build_result, load_stage_outputs

SYN = ROOT / "data/synthetic/synthetic_tos.mp4"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not SYN.exists(), reason="run scripts/make_synthetic.py first")]


@pytest.fixture(scope="module")
def run():
    cfg = load_config()
    result = analyze(SYN, cfg)
    label = json.loads(SYN.with_suffix(".json").read_text())
    return cfg, result, label


def test_scene_metadata_shape(run):
    _, r, _ = run
    assert r["scene_count"] == len(r["scenes"]) >= 1
    assert r["scenes"][0]["start_seconds"] == 0.0
    assert r["scenes"][-1]["end_seconds"] == pytest.approx(r["duration"], abs=0.05)
    for a, b in zip(r["scenes"], r["scenes"][1:]):
        assert a["end_seconds"] == b["start_seconds"]


def test_groups_shots_into_scenes(run):
    _, r, _ = run
    assert r["shot_count"] > 2 * r["scene_count"], "scene layer should merge many shots"


def test_beats_all_shots_baseline(run):
    cfg, r, label = run
    info = probe(SYN)
    stage = load_stage_outputs(cfg, info)
    ours = evaluate(r["boundaries"], label["boundaries"], r["duration"])
    base = evaluate(grouping.baseline_all_shots(stage["shots"]), label["boundaries"], r["duration"])
    assert ours["by_tolerance"]["2"]["f1"] > base["by_tolerance"]["2"]["f1"]
    assert ours["by_tolerance"]["2"]["recall"] >= 0.75


def test_regroup_uses_cache(run):
    """Changing a grouping threshold must not recompute shots/embeddings (and must not
    overwrite the stored analysis)."""
    import time
    cfg, r, _ = run
    info = probe(SYN)
    t = time.perf_counter()
    stage = load_stage_outputs(cfg, info)
    alt = build_result(info, stage, cfg.grouping.replace(abs_threshold=0.6), {})
    assert time.perf_counter() - t < 5.0
    assert alt["shot_count"] == r["shot_count"]


def test_clip_export(run, tmp_path):
    cfg, r, _ = run
    picked = [r["scenes"][0], r["scenes"][3]]
    s = export_clips(SYN, picked, tmp_path, r["fps"], cfg.export)
    assert len(s["files"]) == 2 and s["reencoded"] == 2
    for f, sc in zip(s["files"], picked):
        n_frames = sum(1 for _ in iter_frames(f, 32, 18))
        assert n_frames == pytest.approx(sc["duration_seconds"] * r["fps"], abs=1.0)
