import json

import numpy as np
import pytest

from sceneseen.corrections import CorrectionError, CorrectionStore, merge, split
from sceneseen.features import sample_frame_indices
from sceneseen.ground_truth import (LabelError, find_video, labelled_videos, load_or_create_split, make_split, save_label,
                                    validate_label)
from sceneseen.media import scaled_size
from sceneseen.shots import cuts_to_shots, merge_short_shots, predictions_to_cuts


# ---- shots

def test_predictions_to_cuts_hard_cut_and_dissolve():
    p = np.zeros(100)
    p[19] = 0.9            # hard cut: frame 19 is last of shot 1 -> new shot at 20
    p[50:56] = 0.8         # dissolve 50..55 -> new shot in the middle
    assert predictions_to_cuts(p, 0.5) == [20, 53]


def test_predictions_to_cuts_ignores_edges():
    p = np.zeros(10)
    p[-1] = 1.0
    assert predictions_to_cuts(p, 0.5) == []


def test_merge_short_shots():
    assert merge_short_shots([10, 12, 30, 97], 100, 5) == [10, 30]
    assert merge_short_shots([10, 95], 100, 5) == [10, 95]
    assert merge_short_shots([], 100, 5) == []
    assert merge_short_shots([3], 100, 5) == []


def test_cuts_to_shots_times():
    shots = cuts_to_shots([25, 50], 100, 25.0, 4.02)
    assert [(s.start, s.end) for s in shots] == [(0.0, 1.0), (1.0, 2.0), (2.0, 4.02)]
    assert [s.index for s in shots] == [0, 1, 2]


def test_sample_frame_indices_inside_shot():
    shots = cuts_to_shots([10, 11], 50, 25.0, 2.0)
    idx = sample_frame_indices(shots, (0.15, 0.5, 0.85))
    for s, row in zip(shots, idx):
        assert all(s.start_frame <= i < s.end_frame for i in row)


def test_scaled_size_even():
    assert scaled_size(1280, 534, 224) == (538, 224)
    w, h = scaled_size(1080, 1920, 224)
    assert w == 224 and h % 2 == 0


# ---- corrections

def test_merge_and_split():
    b = [10.0, 20.0, 30.0]
    assert merge(b, 1) == [10.0, 30.0]
    with pytest.raises(CorrectionError):
        merge(b, 3)
    nb, at = split(b, 14.6, 40.0, cut_times=[12.0, 15.0])
    assert at == 15.0 and nb == [10.0, 15.0, 20.0, 30.0]
    _, at = split(b, 13.6, 40.0, cut_times=[12.0, 14.5])   # late press: snaps back 1.6 s
    assert at == 12.0                                       # 14.5 is > 0.5 s ahead
    _, at = split(b, 13.6, 40.0, cut_times=[11.5])          # 2.1 s back: outside window
    assert at == 13.6
    _, at = split(b, 13.6, 40.0, cut_times=[12.1])
    assert at == 12.1
    _, at = split(b, 13.6, 40.0, cut_times=[14.5])          # >0.5 s ahead: no snap
    assert at == 13.6
    with pytest.raises(CorrectionError):
        split(b, 20.1, 40.0)
    with pytest.raises(CorrectionError):
        split(b, 45.0, 40.0)


def test_correction_store(tmp_path):
    st = CorrectionStore(tmp_path)
    assert st.load() is None
    st.save([10.0], {"op": "merge", "scene": 2}, predicted=[10.0, 20.0])
    d = st.save([10.0, 15.0], {"op": "split", "at": 15.0}, predicted=[10.0, 20.0])
    assert d["boundaries"] == [10.0, 15.0] and len(d["log"]) == 2
    assert d["predicted_boundaries"] == [10.0, 20.0]
    st.reset()
    assert st.load() is None


# ---- labels

def test_validate_label():
    validate_label({"video": "a.mp4", "boundaries": [1.0, 2.5]})
    validate_label({"video": "a.mp4", "boundaries": []})
    for bad in ({"video": "a.mp4"}, {"video": "a.mp4", "boundaries": [3, 2]},
                {"video": "a.mp4", "boundaries": [0, 2]}, {"video": "a.mp4", "boundaries": ["1"]}):
        with pytest.raises(LabelError):
            validate_label(bad)
    with pytest.raises(LabelError):
        validate_label({"video": "a.mp4", "boundaries": [50]}, duration=40)


def test_save_label_roundtrip(tmp_path):
    p = save_label(tmp_path, "clip 01.mp4", [20.0, 5.0], annotator="me")
    d = json.loads(p.read_text())
    assert p.name == "clip 01.json" and d["boundaries"] == [5.0, 20.0]


def test_split_is_deterministic_and_frozen(tmp_path):
    stems = [f"v{i}" for i in range(7)]
    s1, s2 = make_split(stems), make_split(list(reversed(stems)))
    assert s1 == s2 and len(s1["test"]) == 2 and not set(s1["dev"]) & set(s1["test"])
    first = load_or_create_split(tmp_path, stems)
    again = load_or_create_split(tmp_path, stems + ["v_new"])
    assert again["test"] == first["test"]          # test set never changes
    assert "v_new" in again["dev"]                 # new videos go to dev


# ---- cross-platform file names (macOS writes decomposed Unicode, Git/Windows/Linux composed)

NFD_NAME = "\u0627\u0654\u0646\u0627 clip.mp4"   # alef + combining hamza (as macOS stored it)
NFC_NAME = "\u0623\u0646\u0627 clip.mp4"         # precomposed alef-with-hamza


def test_find_video_matches_across_unicode_forms(tmp_path):
    (tmp_path / NFC_NAME).write_bytes(b"x")
    assert find_video(tmp_path, NFD_NAME).exists()


def test_label_and_split_match_across_unicode_forms(tmp_path):
    gt, vids = tmp_path / "gt", tmp_path / "videos"
    gt.mkdir(); vids.mkdir()
    (vids / NFC_NAME).write_bytes(b"x")
    (gt / "\u0623\u0646\u0627 clip.json").write_text(json.dumps({"video": NFD_NAME, "boundaries": [1.0]}), encoding="utf-8")
    (gt / "splits.json").write_text(json.dumps({"dev": ["\u0627\u0654\u0646\u0627 clip"], "test": []}, ensure_ascii=False),
                                    encoding="utf-8")
    items = labelled_videos(gt, vids)
    assert len(items) == 1
    before = (gt / "splits.json").read_bytes()
    sp = load_or_create_split(gt, [s for s, *_ in items])
    assert items[0][0] in sp["dev"]                       # recognised, not treated as a new video
    assert (gt / "splits.json").read_bytes() == before    # and the file was not rewritten


def test_manifest_and_splits_are_not_labels(tmp_path):
    gt, vids = tmp_path / "gt", tmp_path / "videos"
    gt.mkdir(); vids.mkdir()
    (vids / "a.mp4").write_bytes(b"x")
    (gt / "a.json").write_text(json.dumps({"video": "a.mp4", "boundaries": [1.0]}), encoding="utf-8")
    (gt / "videos_manifest.json").write_text(json.dumps({"videos": []}), encoding="utf-8")
    (gt / "splits.json").write_text(json.dumps({"dev": ["a"], "test": []}), encoding="utf-8")
    assert [s for s, *_ in labelled_videos(gt, vids)] == ["a"]
