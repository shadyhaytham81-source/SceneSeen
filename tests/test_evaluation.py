import pytest

from sceneseen.evaluation import aggregate, boundary_prf, coverage_overflow, evaluate, match_boundaries


def test_perfect_prediction():
    r = evaluate([10, 20, 30], [10, 20, 30], 40)
    for t in ("1", "2", "3"):
        assert r["by_tolerance"][t]["f1"] == 1.0
    assert r["coverage"] == pytest.approx(1.0)
    assert r["overflow"] == pytest.approx(0.0)
    assert r["f_co"] == pytest.approx(1.0)


def test_tolerance_window():
    pred, gt = [11.5], [10.0]
    assert boundary_prf(pred, gt, 1.0)["tp"] == 0
    r2 = boundary_prf(pred, gt, 2.0)
    assert r2["tp"] == 1 and r2["mean_abs_error"] == pytest.approx(1.5)


def test_one_to_one_matching():
    # two predictions near one label: only one may count
    r = boundary_prf([9.8, 10.3], [10.0], 1.0)
    assert r["tp"] == 1 and r["fp"] == 1 and r["precision"] == 0.5 and r["recall"] == 1.0


def test_optimal_assignment_beats_greedy():
    # greedy nearest would match 10.9->11 and leave 10 unmatched; optimal matches both
    m = match_boundaries([10.9, 12.0], [10.0, 11.0], tolerance=1.0)
    assert len(m) == 2


def test_empty_cases():
    assert boundary_prf([], [], 2)["f1"] == 1.0  # nothing to find, nothing predicted
    r = boundary_prf([], [10], 2)
    assert r["recall"] == 0.0 and r["fn"] == 1
    r = boundary_prf([5], [], 2)
    assert r["precision"] == 0.0 and r["fp"] == 1


def test_oversegmentation_hurts_coverage_not_overflow():
    co = coverage_overflow([5, 10, 15, 20, 25, 30, 35], [20], 40)
    assert co["coverage"] < 0.5
    assert co["overflow"] == pytest.approx(0.0)


def test_undersegmentation_hurts_overflow_not_coverage():
    co = coverage_overflow([], [10, 20, 30], 40)
    assert co["coverage"] == pytest.approx(1.0)
    assert co["overflow"] > 0.5


def test_count_ratio_and_out_of_range_boundaries_ignored():
    r = evaluate([0, 10, 50], [10, 20], 40)
    assert r["n_pred_boundaries"] == 1
    assert r["count_ratio"] == pytest.approx(2 / 3)


def test_aggregate_micro_and_macro():
    a = evaluate([10], [10], 30)
    b = evaluate([], [10, 20], 30)
    agg = aggregate([a, b])
    assert agg["videos"] == 2
    t = agg["by_tolerance"]["2"]
    assert t["macro_f1"] == pytest.approx(0.5)
    assert t["micro_precision"] == pytest.approx(1.0)
    assert t["micro_recall"] == pytest.approx(1 / 3)
