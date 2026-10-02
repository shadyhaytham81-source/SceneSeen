import numpy as np
import pytest

from sceneseen import grouping
from sceneseen.config import GroupingConfig
from sceneseen.shots import Shot


def make_shots(durations, fps=25.0):
    shots, f = [], 0
    for i, d in enumerate(durations):
        n = int(d * fps)
        shots.append(Shot(i, f, f + n, f / fps, (f + n) / fps))
        f += n
    return shots


def features_from_labels(labels, dim=16, seed=0):
    """One random unit vector per visual 'setup' label (A, B, C...). Shots with the
    same label get near-identical features; different labels are near-orthogonal."""
    rng = np.random.default_rng(seed)
    protos = {}
    clip = np.zeros((len(labels), 3, dim), np.float32)
    color = np.zeros((len(labels), 3, 8), np.float32)
    for i, lab in enumerate(labels):
        if lab not in protos:
            v = rng.normal(size=dim)
            h = rng.random(8) ** 4
            protos[lab] = (v / np.linalg.norm(v), h / h.sum())
        v, h = protos[lab]
        for j in range(3):
            noise = rng.normal(scale=0.03, size=dim)
            e = v + noise
            clip[i, j] = e / np.linalg.norm(e)
            color[i, j] = h
    return clip, color


CFG = GroupingConfig(min_scene_seconds=3.0)


def boundaries_for(labels, durations=None, cfg=CFG):
    durations = durations or [4.0] * len(labels)
    shots = make_shots(durations)
    clip, color = features_from_labels(labels)
    dur = shots[-1].end
    return grouping.group_shots(shots, clip, color, dur, cfg), shots


def test_dialogue_abab_stays_one_scene():
    # restaurant: wide W, Ahmed A, Sara B alternate; then home at night H, I
    labels = ["W", "A", "B", "A", "B", "A", "W", "H", "I", "H", "I", "H"]
    res, shots = boundaries_for(labels)
    assert res["boundaries"] == [shots[7].start]
    assert len(res["scenes"]) == 2


def test_adjacent_only_baseline_oversegments_dialogue():
    labels = ["W", "A", "B", "A", "B", "A", "W", "H", "I", "H", "I", "H"]
    shots = make_shots([4.0] * len(labels))
    clip, color = features_from_labels(labels)
    adj = grouping.baseline_adjacent(shots, clip, color, shots[-1].end, CFG)
    assert len(adj) > 1  # every A<->B cut looks like a change without context


def test_three_scenes():
    labels = ["A", "A", "B", "A", "B", "C", "D", "C", "D", "E", "E", "F", "E", "F"]
    res, shots = boundaries_for(labels)
    assert res["boundaries"] == [shots[5].start, shots[9].start]


def test_min_scene_duration_suppresses_short_scene():
    labels = ["A", "B", "A", "B", "X", "C", "D", "C", "D"]
    durations = [4, 4, 4, 4, 1.0, 4, 4, 4, 4]
    res, shots = boundaries_for(labels, durations, CFG.replace(min_scene_seconds=3.0))
    for a, b in zip([0.0] + res["boundaries"], res["boundaries"] + [shots[-1].end]):
        assert b - a >= 3.0


def test_single_shot_video():
    shots = make_shots([10.0])
    clip, color = features_from_labels(["A"])
    res = grouping.group_shots(shots, clip, color, 10.0, CFG)
    assert res["boundaries"] == [] and len(res["scenes"]) == 1


def test_uniform_video_has_no_boundaries():
    labels = ["A"] * 10
    res, _ = boundaries_for(labels)
    assert res["boundaries"] == []


def test_decisions_are_recorded_for_every_cut():
    labels = ["A", "B", "A", "C", "D", "C"]
    res, _ = boundaries_for(labels)
    assert len(res["cuts"]) == len(labels) - 1
    assert {c["decision"] for c in res["cuts"]} <= {"boundary", "below_abs", "below_depth", "min_duration"}


def test_depth_scores():
    d = grouping.depth_scores(np.array([0.1, 0.2, 0.9, 0.3, 0.1]))
    assert d[2] == pytest.approx((0.9 - 0.1) + (0.9 - 0.1))
    assert d[0] == pytest.approx(0.0)


def test_window_seconds_blocks_distant_links():
    # A ... A separated by a long C shot: with a short time window, the far A cannot bridge the cut
    labels = ["A", "B", "C", "A"]
    durations = [4, 4, 60, 4]
    shots = make_shots(durations)
    clip, color = features_from_labels(labels)
    S = grouping.similarity_matrix(clip, color, CFG)
    near = grouping.cut_scores(S, shots, CFG.replace(window_seconds=10))
    far = grouping.cut_scores(S, shots, CFG.replace(window_seconds=1000))
    assert far[1].coherence > near[1].coherence


def test_boundaries_to_scenes_covers_video():
    scenes = grouping.boundaries_to_scenes([10.0, 25.5], 60.0)
    assert [s["start_seconds"] for s in scenes] == [0.0, 10.0, 25.5]
    assert scenes[-1]["end_seconds"] == 60.0
    assert sum(s["duration_seconds"] for s in scenes) == pytest.approx(60.0)


def test_baseline_all_shots():
    shots = make_shots([2, 3, 4])
    assert grouping.baseline_all_shots(shots) == [2.0, 5.0]


def test_config_replace_rejects_unknown():
    with pytest.raises(ValueError):
        CFG.replace(nope=1)


def test_window_depth_handles_adjacent_boundaries():
    # two consecutive scene changes (0.7, 0.72) between ordinary cuts (0.2)
    s = np.array([0.2, 0.2, 0.7, 0.72, 0.2, 0.2])
    climb = grouping.depth_scores(s)
    win = grouping.window_depth_scores(s, radius=3)
    assert climb[2] < 0.6          # climb stops at the higher neighbour
    assert win[2] > 0.9            # window still sees the within-scene cuts
    assert win[3] == pytest.approx(climb[3])


def test_unknown_depth_mode_rejected():
    labels = ["A", "B", "A", "C"]
    with pytest.raises(ValueError):
        boundaries_for(labels, cfg=CFG.replace(depth_mode="nope"))
