"""Extended anomaly analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import (
    anomaly_rate,
    batch_compute_severity,
    compute_severity,
    consecutive_anomaly_runs,
    ewma_smooth,
    flag_anomaly_rate,
    iqr_flag,
    zscore_flag,
)


def test_zscore_normal() -> None:
    assert not zscore_flag(10.0, mean=10.0, std=2.0, threshold=3.0)


def test_zscore_outlier() -> None:
    assert zscore_flag(20.0, mean=10.0, std=1.0, threshold=3.0)


def test_zscore_zero_std() -> None:
    assert not zscore_flag(5.0, mean=5.0, std=0.0)


def test_iqr_within_fence() -> None:
    assert not iqr_flag(10.0, q1=8.0, q3=12.0)


def test_iqr_outside_fence() -> None:
    assert iqr_flag(30.0, q1=8.0, q3=12.0)


def test_iqr_below_fence() -> None:
    assert iqr_flag(-5.0, q1=8.0, q3=12.0)


def test_compute_severity_none() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_critical() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(100.0, ref)
    assert result["severity"] == "critical"
    assert result["z_flag"] is True
    assert result["iqr_flag"] is True


@pytest.mark.parametrize("value,expected", [(10.0, "none"), (100.0, "critical")])
def test_compute_severity_parametrized(value, expected) -> None:
    ref = list(np.random.default_rng(42).normal(10, 1, 100))
    result = compute_severity(value, ref)
    assert result["severity"] == expected


def test_compute_severity_returns_required_keys() -> None:
    ref = list(np.random.default_rng(0).normal(5, 1, 50))
    result = compute_severity(5.0, ref)
    assert "z_flag" in result
    assert "iqr_flag" in result
    assert "severity" in result


def test_zscore_exactly_at_threshold() -> None:
    assert not zscore_flag(13.0, mean=10.0, std=1.0, threshold=3.0)


@pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0, 4.0])
def test_zscore_various_thresholds(threshold) -> None:
    assert zscore_flag(10.0 + threshold * 1.5, mean=10.0, std=1.0, threshold=threshold)


def test_iqr_exactly_at_fence() -> None:
    assert not iqr_flag(20.0, q1=8.0, q3=12.0, k=2.0)


def test_compute_severity_warning_only_one_flag() -> None:
    ref = list(np.random.default_rng(7).normal(10, 1, 200))
    # Use a modest outlier that might trigger only one of the two tests
    result = compute_severity(13.5, ref)
    assert result["severity"] in ("none", "warning", "critical")


def test_batch_compute_severity_basic() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    results = batch_compute_severity([10.0, 100.0], ref)
    assert len(results) == 2
    assert results[0]["severity"] == "none"
    assert results[1]["severity"] == "critical"


def test_batch_compute_severity_includes_value_key() -> None:
    ref = list(np.random.default_rng(2).normal(10, 1, 100))
    results = batch_compute_severity([10.0, 11.0], ref)
    assert all("value" in r for r in results)
    assert results[0]["value"] == 10.0


def test_batch_compute_severity_small_reference() -> None:
    results = batch_compute_severity([5.0, 6.0], [1.0, 2.0, 3.0])
    assert all(r["severity"] == "none" for r in results)


def test_batch_compute_severity_empty_input() -> None:
    ref = list(np.random.default_rng(3).normal(10, 1, 100))
    results = batch_compute_severity([], ref)
    assert results == []


@pytest.mark.parametrize(
    "flagged,total,expected",
    [
        ([], 0, 0.0),
        ([{"severity": "none"}], 1, 0.0),
        ([{"severity": "warning"}], 1, 1.0),
        ([{"severity": "critical"}, {"severity": "none"}], 2, 0.5),
    ],
)
def test_anomaly_rate_parametrized(flagged, total, expected) -> None:
    assert anomaly_rate(flagged) == pytest.approx(expected)


def test_top_anomalies_returns_critical_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 10.0},
        {"severity": "critical", "value": 100.0},
        {"severity": "warning", "value": 15.0},
    ]
    result = top_anomalies(data, n=3)
    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "warning"


def test_top_anomalies_respects_n() -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "critical", "value": float(i)} for i in range(20)]
    result = top_anomalies(data, n=5)
    assert len(result) == 5


def test_top_anomalies_empty() -> None:
    from app.anomaly import top_anomalies

    assert top_anomalies([], n=5) == []


def test_top_anomalies_custom_order() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 1.0},
        {"severity": "warning", "value": 2.0},
    ]
    result = top_anomalies(data, n=2, severity_order=["warning", "critical", "none"])
    assert result[0]["severity"] == "warning"


def test_compute_percentile_bounds_basic() -> None:
    from app.anomaly import compute_percentile_bounds

    ref = list(range(1, 101))  # 1..100
    bounds = compute_percentile_bounds(ref, lower_pct=1.0, upper_pct=99.0)
    assert bounds["lower"] < bounds["upper"]
    assert bounds["median"] == pytest.approx(50.5, rel=1e-2)


def test_compute_percentile_bounds_keys() -> None:
    from app.anomaly import compute_percentile_bounds

    bounds = compute_percentile_bounds([5.0] * 20)
    assert set(bounds.keys()) == {"lower", "upper", "median", "mean"}


def test_compute_percentile_bounds_empty_raises() -> None:
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError, match="non-empty"):
        compute_percentile_bounds([])


def test_compute_percentile_bounds_invalid_pct_raises() -> None:
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError):
        compute_percentile_bounds([1.0, 2.0], lower_pct=80.0, upper_pct=20.0)


@pytest.mark.parametrize("lower,upper", [(0, 100), (5, 95), (10, 90), (25, 75)])
def test_compute_percentile_bounds_parametrized(lower, upper) -> None:
    from app.anomaly import compute_percentile_bounds

    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    bounds = compute_percentile_bounds(ref, lower_pct=lower, upper_pct=upper)
    assert bounds["lower"] <= bounds["upper"]


def test_classify_consumption_low() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(3.0, low_threshold=5.0, high_threshold=15.0) == "low"


def test_classify_consumption_normal() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(10.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_high() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(20.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_boundary_low() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(5.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_boundary_high() -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(15.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_invalid_thresholds() -> None:
    from app.anomaly import classify_consumption

    with pytest.raises(ValueError):
        classify_consumption(10.0, low_threshold=20.0, high_threshold=10.0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "low"),
        (4.9, "low"),
        (5.0, "normal"),
        (14.9, "normal"),
        (15.0, "high"),
        (100.0, "high"),
    ],
)
def test_classify_consumption_parametrized(value, expected) -> None:
    from app.anomaly import classify_consumption

    assert classify_consumption(value, low_threshold=5.0, high_threshold=15.0) == expected


def test_compute_z_score_basic() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(10.0, mean=10.0, std=1.0) == pytest.approx(0.0)


def test_compute_z_score_positive() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(13.0, mean=10.0, std=1.0) == pytest.approx(3.0)


def test_compute_z_score_negative() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(7.0, mean=10.0, std=1.0) == pytest.approx(-3.0)


def test_compute_z_score_zero_std() -> None:
    from app.anomaly import compute_z_score

    assert compute_z_score(5.0, mean=5.0, std=0.0) == pytest.approx(0.0)


def test_flag_z_score_outliers_finds_outlier() -> None:
    from app.anomaly import flag_z_score_outliers

    data = [10.0] * 50 + [1000.0]
    outliers = flag_z_score_outliers(data)
    assert 50 in outliers


def test_flag_z_score_outliers_none_on_flat() -> None:
    from app.anomaly import flag_z_score_outliers

    data = [5.0] * 20
    assert flag_z_score_outliers(data) == []


def test_flag_z_score_outliers_empty_raises() -> None:
    from app.anomaly import flag_z_score_outliers

    with pytest.raises(ValueError):
        flag_z_score_outliers([])


def test_flag_z_score_outliers_bad_threshold_raises() -> None:
    from app.anomaly import flag_z_score_outliers

    with pytest.raises(ValueError):
        flag_z_score_outliers([1.0, 2.0], threshold=0.0)


@pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0])
def test_flag_z_score_outliers_custom_threshold(threshold) -> None:
    from app.anomaly import flag_z_score_outliers

    data = [10.0] * 40 + [10000.0]
    result = flag_z_score_outliers(data, threshold=threshold)
    assert len(result) >= 1


def test_top_anomalies_returns_most_severe_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 5.0},
        {"severity": "critical", "value": 100.0},
        {"severity": "warning", "value": 50.0},
    ]
    result = top_anomalies(data, n=3)
    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "warning"


def test_top_anomalies_limits_to_n() -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "critical", "value": float(i)} for i in range(20)]
    result = top_anomalies(data, n=5)
    assert len(result) == 5


def test_top_anomalies_empty_input() -> None:
    from app.anomaly import top_anomalies

    assert top_anomalies([]) == []


def test_top_anomalies_custom_order_warning_first() -> None:
    from app.anomaly import top_anomalies

    data = [
        {"severity": "warning", "value": 5.0},
        {"severity": "critical", "value": 10.0},
    ]
    result = top_anomalies(data, n=2, severity_order=["warning", "critical", "none"])
    assert result[0]["severity"] == "warning"


@pytest.mark.parametrize("n", [1, 5, 10])
def test_top_anomalies_n_parametrized(n) -> None:
    from app.anomaly import top_anomalies

    data = [{"severity": "warning", "value": float(i)} for i in range(15)]
    result = top_anomalies(data, n=n)
    assert len(result) <= n


def test_flag_anomaly_rate_all_anomalies() -> None:
    assert flag_anomaly_rate([True, True, True]) == pytest.approx(1.0)


def test_flag_anomaly_rate_no_anomalies() -> None:
    assert flag_anomaly_rate([False, False, False]) == pytest.approx(0.0)


def test_flag_anomaly_rate_mixed() -> None:
    assert flag_anomaly_rate([True, False, True, False]) == pytest.approx(0.5)


def test_flag_anomaly_rate_empty() -> None:
    assert flag_anomaly_rate([]) == 0.0


def test_consecutive_anomaly_runs_single_run() -> None:
    flags = [False, True, True, True, False]
    runs = consecutive_anomaly_runs(flags)
    assert runs == [(1, 3)]


def test_consecutive_anomaly_runs_multiple() -> None:
    flags = [True, False, True, True]
    runs = consecutive_anomaly_runs(flags)
    assert (0, 0) in runs
    assert (2, 3) in runs


def test_consecutive_anomaly_runs_none() -> None:
    flags = [False, False, False]
    assert consecutive_anomaly_runs(flags) == []


def test_consecutive_anomaly_runs_all_true() -> None:
    flags = [True, True, True]
    runs = consecutive_anomaly_runs(flags)
    assert runs == [(0, 2)]


@pytest.mark.parametrize("rate,expected", [(1.0, True), (0.0, False)])
def test_flag_anomaly_rate_boundary(rate, expected) -> None:
    n = 10
    flags = [True] * int(n * rate) + [False] * (n - int(n * rate))
    result = flag_anomaly_rate(flags)
    assert (result == pytest.approx(1.0)) == expected


@pytest.mark.parametrize("value,mean,std,threshold,expected", [
    (10.0, 10.0, 2.0, 3.0, False),    # within threshold
    (19.0, 10.0, 2.0, 3.0, True),     # 4.5 stdev away
    (10.0, 10.0, 0.0, 3.0, False),    # zero std -> not anomaly
    (100.0, 10.0, 1.0, 3.0, True),    # far outlier
])
def test_zscore_flag_parametrized(value: float, mean: float, std: float, threshold: float, expected: bool) -> None:
    assert zscore_flag(value, mean=mean, std=std, threshold=threshold) == expected


@pytest.mark.parametrize("value,q1,q3,expected", [
    (10.0, 8.0, 12.0, False),    # within fence
    (30.0, 8.0, 12.0, True),     # above upper fence
    (-5.0, 8.0, 12.0, True),     # below lower fence
    (12.0, 8.0, 12.0, False),    # exactly at q3
])
def test_iqr_flag_parametrized(value: float, q1: float, q3: float, expected: bool) -> None:
    assert iqr_flag(value, q1=q1, q3=q3) == expected


def test_compute_severity_small_reference() -> None:
    result = compute_severity(10.0, reference=[5.0, 6.0])
    assert "severity" in result


def test_batch_compute_severity_empty() -> None:
    result = batch_compute_severity([], reference=[10.0, 11.0, 12.0, 9.0, 10.5])
    assert result == []


def test_anomaly_rate_all_critical() -> None:
    severities = [{"severity": "critical", "z_flagged": True, "iqr_flagged": True} for _ in range(10)]
    assert anomaly_rate(severities) == pytest.approx(1.0)


def test_anomaly_rate_all_none() -> None:
    severities = [{"severity": "none", "z_flagged": False, "iqr_flagged": False} for _ in range(10)]
    assert anomaly_rate(severities) == pytest.approx(0.0)


@pytest.mark.parametrize("n_values,n_ref", [(1, 10), (5, 20), (100, 50)])
def test_batch_compute_severity_sizes(n_values: int, n_ref: int) -> None:
    rng = np.random.default_rng(99)
    ref = list(rng.normal(10, 1, n_ref).tolist())
    vals = list(rng.normal(10, 1, n_values).tolist())
    results = batch_compute_severity(vals, reference=ref)
    assert len(results) == n_values
    for r in results:
        assert r["severity"] in ("none", "warning", "critical")


def test_anomaly_rate_mixed_severities() -> None:
    severities = [
        {"severity": "critical"},
        {"severity": "warning"},
        {"severity": "none"},
        {"severity": "critical"},
    ]
    rate = anomaly_rate(severities)
    assert rate == pytest.approx(0.75, rel=1e-4)


def test_compute_severity_warning_zone() -> None:
    ref = [10.0] * 200
    # value exactly at 4 std from mean (std ~0 so any deviation is flagged)
    result = compute_severity(10.5, ref, z_threshold=3.0)
    assert "severity" in result



def test_ewma_smooth_length() -> None:
    result = ewma_smooth([1.0, 2.0, 3.0, 4.0], alpha=0.3)
    assert len(result) == 4


def test_ewma_smooth_first_value_unchanged() -> None:
    values = [10.0, 20.0, 30.0]
    result = ewma_smooth(values, alpha=0.5)
    assert result[0] == pytest.approx(10.0)


def test_ewma_smooth_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        ewma_smooth([])


def test_ewma_smooth_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ewma_smooth([1.0, 2.0], alpha=0.0)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_ewma_smooth_monotone_input(alpha: float) -> None:
    values = [float(i) for i in range(1, 11)]
    result = ewma_smooth(values, alpha=alpha)
    assert len(result) == len(values)
    assert all(isinstance(v, float) for v in result)


# Tests for anomaly_summary and rolling_anomaly_flag
from app.anomaly import anomaly_summary, rolling_anomaly_flag


def test_anomaly_summary_empty() -> None:
    assert anomaly_summary([]) == {"none": 0, "warning": 0, "critical": 0}


def test_anomaly_summary_counts() -> None:
    sevs = [
        {"severity": "none"},
        {"severity": "warning"},
        {"severity": "critical"},
        {"severity": "critical"},
    ]
    s = anomaly_summary(sevs)
    assert s["none"] == 1
    assert s["warning"] == 1
    assert s["critical"] == 2


def test_anomaly_summary_all_none() -> None:
    sevs = [{"severity": "none"}] * 5
    s = anomaly_summary(sevs)
    assert s["none"] == 5
    assert s["warning"] == 0
    assert s["critical"] == 0


def test_rolling_anomaly_flag_length() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 3.0, 2.0, 1.0]
    flags = rolling_anomaly_flag(values, window=5)
    assert len(flags) == len(values)


def test_rolling_anomaly_flag_empty_raises() -> None:
    with pytest.raises(ValueError):
        rolling_anomaly_flag([])


def test_rolling_anomaly_flag_small_window_raises() -> None:
    with pytest.raises(ValueError):
        rolling_anomaly_flag([1.0, 2.0], window=1)


def test_rolling_anomaly_flag_constant_no_flags() -> None:
    values = [5.0] * 20
    flags = rolling_anomaly_flag(values, window=5)
    assert not any(flags)


def test_rolling_anomaly_flag_outlier_flagged() -> None:
    values = [1.0] * 10 + [1000.0] + [1.0] * 10
    flags = rolling_anomaly_flag(values, window=5)
    assert flags[10] is True


@pytest.mark.parametrize("n", [5, 10, 20])
def test_rolling_anomaly_flag_length_param(n: int) -> None:
    values = list(range(n))
    flags = rolling_anomaly_flag(values, window=3)
    assert len(flags) == n
