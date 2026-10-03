"""Unique Shots (repeated-shot grouping) on synthetic shots with controlled similarity."""
import json

import numpy as np
import pytest

from sceneseen import similarity as sim
from sceneseen import unique as U
from sceneseen.config import UniqueShotsConfig
from sceneseen.shots import Shot

CFG = UniqueShotsConfig()
D, BITS, BINS = 64, 64, 16


class Maker:
    """Builds shots from named camera set-ups.

    'A'      -> a camera set-up: its own content (CLIP), framing (pHash) and palette (colour)
    'A~'     -> near duplicate of A (actor moved a little, slight light change)
    'A/angle' -> same actor/content as A filmed from another angle: CLIP stays high (~0.9),
                 palette stays, framing is unrelated
    'A/room'  -> same room as A, different framing and subject: CLIP ~0.8, palette similar
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.protos = {}

    def _proto(self, name):
        if name not in self.protos:
            v = self.rng.normal(size=D)
            h = self.rng.random(BINS) ** 3
            self.protos[name] = (v / np.linalg.norm(v), self.rng.random(BITS) > 0.5, h / h.sum())
        return self.protos[name]

    def shot(self, spec):
        base, _, variant = spec.partition("/")
        near = base.endswith("~")
        v, bits, hist = self._proto(base.rstrip("~"))
        noise = 0.02 if not near else 0.035
        if variant:
            other = self.rng.normal(size=D)
            other -= other @ v * v
            other /= np.linalg.norm(other)
            cos = 0.90 if variant == "angle" else 0.80
            v = cos * v + np.sqrt(1 - cos ** 2) * other
            bits = bits.copy()
            bits[self.rng.choice(BITS, size=BITS // 2, replace=False)] ^= True   # framing agrees only by chance
            alt = self.rng.random(BINS) ** 3
            hist = 0.8 * hist + 0.2 * alt / alt.sum()                            # background changes with the angle
        else:
            bits = bits.copy()
            flip = self.rng.choice(BITS, size=6 if near else 2, replace=False)
            bits[flip] = ~bits[flip]
        frames = np.stack([v + self.rng.normal(scale=noise, size=D) / np.sqrt(D) * 4 for _ in range(3)])
        frames /= np.linalg.norm(frames, axis=1, keepdims=True)
        h = hist + self.rng.random(BINS) * (0.004 if not variant else 0.01)
        return frames.astype(np.float32), bits, np.tile(h / h.sum(), (3, 1)).astype(np.float32)

    def build(self, specs, durations=None, sharpness=None):
        parts = [self.shot(s) for s in specs]
        n = len(specs)
        durations = durations or [3.0] * n
        shots, t = [], 0.0
        for i, d in enumerate(durations):
            shots.append(Shot(i, int(t * 25), int((t + d) * 25), t, t + d))
            t += d
        stats = {"phash": np.stack([p[1] for p in parts]),
                 "sharpness": np.array(sharpness or [100.0] * n, np.float32),
                 "brightness": np.full(n, 0.45, np.float32), "has_thumb": np.ones(n, bool)}
        scenes = [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": t, "shot_start": 0, "shot_end": n - 1}]
        return shots, scenes, np.stack([p[0] for p in parts]), np.stack([p[2] for p in parts]), stats


def groups_of(specs, **kw):
    shots, scenes, clip, color, stats = Maker().build(specs, **kw)
    rec = U.unique_shots(shots, scenes, clip, color, stats, CFG)
    return rec, [sorted(g["shot_ids"]) for g in rec["scenes"][0]["unique"]]


def test_cosine_similarity_definition():
    a, b = np.array([3.0, 4.0, 0.0]), np.array([4.0, 3.0, 0.0])
    clip = np.stack([np.tile(a, (2, 1)), np.tile(b, (2, 1))])
    expected = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    assert sim.clip_mean_similarity(clip)[0, 1] == pytest.approx(expected)
    fs = sim.clip_frame_similarity(clip)
    assert fs.shape == (2, 2, 2, 2) and fs[0, 1, 0, 0] == pytest.approx(expected)
    assert sim.clip_matched_similarity(fs)[0, 0] == pytest.approx(1.0)


def test_identical_shots_are_one_group():
    rec, g = groups_of(["A", "A", "A"])
    assert g == [[0, 1, 2]] and rec["summary"]["unique_shots"] == 1 and rec["summary"]["repeated_shots"] == 2


def test_near_duplicates_are_grouped():
    _, g = groups_of(["A", "A~", "A~"])
    assert g == [[0, 1, 2]]


def test_same_actor_different_angle_is_not_a_duplicate():
    _, g = groups_of(["A", "A/angle"])
    assert g == [[0], [1]]


def test_same_room_different_framing_is_not_a_duplicate():
    _, g = groups_of(["A", "A/room"])
    assert g == [[0], [1]]


def test_abab_dialogue_groups_each_side():
    rec, g = groups_of(["A", "B", "A", "B", "A", "B"])
    assert g == [[0, 2, 4], [1, 3, 5]]
    sc = rec["scenes"][0]
    assert (sc["original_shots"], sc["unique_shots"], sc["repeated_shots"]) == (6, 2, 4)
    assert sc["reduction"] == pytest.approx(4 / 6, abs=1e-3)


def test_example_from_spec_a_b_a_a_c_b():
    _, g = groups_of(["A", "B", "A", "A", "C", "B"])
    assert g == [[0, 2, 3], [1, 5], [4]]


def test_no_duplicates():
    rec, g = groups_of(["A", "B", "C", "D"])
    assert g == [[0], [1], [2], [3]] and rec["summary"]["reduction"] == 0.0
    assert all(x["duplicate_shot_ids"] == [] for x in rec["scenes"][0]["unique"])


def test_all_duplicates():
    rec, g = groups_of(["A"] * 7)
    assert g == [list(range(7))] and rec["summary"]["unique_shots"] == 1


def test_representative_prefers_sharp_long_shot():
    rec, _ = groups_of(["A", "A", "A"], durations=[1.0, 6.0, 1.0], sharpness=[20.0, 400.0, 20.0])
    grp = rec["scenes"][0]["unique"][0]
    assert grp["representative_shot_id"] == 1
    assert sorted(grp["duplicate_shot_ids"]) == [0, 2]
    assert grp["occurrence_count"] == 3


def test_traceability_every_shot_kept_exactly_once():
    specs = ["A", "B", "A", "C", "B", "A~", "D", "C"]
    rec, _ = groups_of(specs)
    seen = [sid for g in rec["scenes"][0]["unique"] for sid in g["shot_ids"]]
    assert sorted(seen) == list(range(len(specs)))          # nothing deleted, nothing duplicated
    for g in rec["scenes"][0]["unique"]:
        assert g["representative_shot_id"] in g["shot_ids"]
        assert set(g["duplicate_shot_ids"]) == set(g["shot_ids"]) - {g["representative_shot_id"]}
        assert len(g["occurrences"]) == g["occurrence_count"]
        for o in g["occurrences"]:
            assert {"shot_id", "start", "end", "similarity_to_representative"} <= set(o)
    json.dumps(rec)                                          # serialisable


def test_groups_never_cross_scene_boundaries():
    m = Maker()
    shots, _, clip, color, stats = m.build(["A", "B", "A", "A", "B", "A"])
    scenes = [{"scene_id": 1, "start_seconds": 0, "end_seconds": 9, "shot_start": 0, "shot_end": 2},
              {"scene_id": 2, "start_seconds": 9, "end_seconds": 18, "shot_start": 3, "shot_end": 5}]
    rec = U.unique_shots(shots, scenes, clip, color, stats, CFG)
    assert [sorted(g["shot_ids"]) for g in rec["scenes"][0]["unique"]] == [[0, 2], [1]]
    assert [sorted(g["shot_ids"]) for g in rec["scenes"][1]["unique"]] == [[3, 5], [4]]
    assert rec["summary"] == {"original_shots": 6, "unique_shots": 4, "repeated_shots": 2,
                              "reduction": pytest.approx(1 / 3, abs=1e-3), "scenes": 2}


def test_uncertain_pairs_are_reported_not_merged():
    rec, g = groups_of(["A", "A/angle"])
    assert g == [[0], [1]]
    rel = rec["scenes"][0]["unique"][0]["possible_duplicates_of"]
    assert rel and rel[0]["unique_shot_id"] == "S01-U02"
    assert CFG.different_threshold <= rel[0]["max_similarity"] < CFG.duplicate_threshold


def test_unique_shots_do_not_touch_segmentation_inputs():
    shots, scenes, clip, color, stats = Maker().build(["A", "B", "A"])
    before = (clip.copy(), color.copy(), json.dumps(scenes), [s.to_dict() for s in shots])
    U.unique_shots(shots, scenes, clip, color, stats, CFG)
    assert np.array_equal(clip, before[0]) and np.array_equal(color, before[1])
    assert json.dumps(scenes) == before[2] and [s.to_dict() for s in shots] == before[3]


def test_cache_roundtrip(tmp_path):
    shots, scenes, clip, color, stats = Maker().build(["A", "B", "A"])
    np.savez_compressed(tmp_path / "visual_k.npz", **stats)
    feats = {"clip": clip, "color": color}
    a = U.unique_shots_cached(tmp_path, "k", tmp_path / "thumbs", shots, scenes, feats, CFG)
    b = U.unique_shots_cached(tmp_path, "k", tmp_path / "thumbs", shots, scenes, feats, CFG)
    assert a["cached"] is False and b["cached"] is True and a["summary"] == b["summary"]
