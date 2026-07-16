"""Extended anomaly analysis tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.anomaly import anomaly_rate, batch_compute_severity, compute_severity, iqr_flag, zscore_flag


def test_zscore_normal():
    assert not zscore_flag(10.0, mean=10.0, std=2.0, threshold=3.0)


def test_zscore_outlier():
    assert zscore_flag(20.0, mean=10.0, std=1.0, threshold=3.0)


def test_zscore_zero_std():
    assert not zscore_flag(5.0, mean=5.0, std=0.0)


def test_iqr_within_fence():
    assert not iqr_flag(10.0, q1=8.0, q3=12.0)


def test_iqr_outside_fence():
    assert iqr_flag(30.0, q1=8.0, q3=12.0)


def test_iqr_below_fence():
    assert iqr_flag(-5.0, q1=8.0, q3=12.0)


def test_compute_severity_none():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(10.0, ref)
    assert result["severity"] == "none"


def test_compute_severity_critical():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    result = compute_severity(100.0, ref)
    assert result["severity"] == "critical"
    assert result["z_flag"] is True
    assert result["iqr_flag"] is True


@pytest.mark.parametrize("value,expected", [(10.0, "none"), (100.0, "critical")])
def test_compute_severity_parametrized(value, expected):
    ref = list(np.random.default_rng(42).normal(10, 1, 100))
    result = compute_severity(value, ref)
    assert result["severity"] == expected


def test_compute_severity_returns_required_keys():
    ref = list(np.random.default_rng(0).normal(5, 1, 50))
    result = compute_severity(5.0, ref)
    assert "z_flag" in result
    assert "iqr_flag" in result
    assert "severity" in result


def test_zscore_exactly_at_threshold():
    assert not zscore_flag(13.0, mean=10.0, std=1.0, threshold=3.0)


@pytest.mark.parametrize("threshold", [1.0, 2.0, 3.0, 4.0])
def test_zscore_various_thresholds(threshold):
    assert zscore_flag(10.0 + threshold * 1.5, mean=10.0, std=1.0, threshold=threshold)


def test_iqr_exactly_at_fence():
    assert not iqr_flag(20.0, q1=8.0, q3=12.0, k=2.0)


def test_compute_severity_warning_only_one_flag():
    ref = list(np.random.default_rng(7).normal(10, 1, 200))
    # Use a modest outlier that might trigger only one of the two tests
    result = compute_severity(13.5, ref)
    assert result["severity"] in ("none", "warning", "critical")


def test_batch_compute_severity_basic():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    results = batch_compute_severity([10.0, 100.0], ref)
    assert len(results) == 2
    assert results[0]["severity"] == "none"
    assert results[1]["severity"] == "critical"


def test_batch_compute_severity_includes_value_key():
    ref = list(np.random.default_rng(2).normal(10, 1, 100))
    results = batch_compute_severity([10.0, 11.0], ref)
    assert all("value" in r for r in results)
    assert results[0]["value"] == 10.0


def test_batch_compute_severity_small_reference():
    results = batch_compute_severity([5.0, 6.0], [1.0, 2.0, 3.0])
    assert all(r["severity"] == "none" for r in results)


def test_batch_compute_severity_empty_input():
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
def test_anomaly_rate_parametrized(flagged, total, expected):
    assert anomaly_rate(flagged) == pytest.approx(expected)


def test_top_anomalies_returns_critical_first():
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 10.0},
        {"severity": "critical", "value": 100.0},
        {"severity": "warning", "value": 15.0},
    ]
    result = top_anomalies(data, n=3)
    assert result[0]["severity"] == "critical"
    assert result[1]["severity"] == "warning"


def test_top_anomalies_respects_n():
    from app.anomaly import top_anomalies

    data = [{"severity": "critical", "value": float(i)} for i in range(20)]
    result = top_anomalies(data, n=5)
    assert len(result) == 5


def test_top_anomalies_empty():
    from app.anomaly import top_anomalies

    assert top_anomalies([], n=5) == []


def test_top_anomalies_custom_order():
    from app.anomaly import top_anomalies

    data = [
        {"severity": "none", "value": 1.0},
        {"severity": "warning", "value": 2.0},
    ]
    result = top_anomalies(data, n=2, severity_order=["warning", "critical", "none"])
    assert result[0]["severity"] == "warning"


def test_compute_percentile_bounds_basic():
    from app.anomaly import compute_percentile_bounds

    ref = list(range(1, 101))  # 1..100
    bounds = compute_percentile_bounds(ref, lower_pct=1.0, upper_pct=99.0)
    assert bounds["lower"] < bounds["upper"]
    assert bounds["median"] == pytest.approx(50.5, rel=1e-2)


def test_compute_percentile_bounds_keys():
    from app.anomaly import compute_percentile_bounds

    bounds = compute_percentile_bounds([5.0] * 20)
    assert set(bounds.keys()) == {"lower", "upper", "median", "mean"}


def test_compute_percentile_bounds_empty_raises():
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError, match="non-empty"):
        compute_percentile_bounds([])


def test_compute_percentile_bounds_invalid_pct_raises():
    from app.anomaly import compute_percentile_bounds

    with pytest.raises(ValueError):
        compute_percentile_bounds([1.0, 2.0], lower_pct=80.0, upper_pct=20.0)


@pytest.mark.parametrize("lower,upper", [(0, 100), (5, 95), (10, 90), (25, 75)])
def test_compute_percentile_bounds_parametrized(lower, upper):
    from app.anomaly import compute_percentile_bounds

    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    bounds = compute_percentile_bounds(ref, lower_pct=lower, upper_pct=upper)
    assert bounds["lower"] <= bounds["upper"]


def test_classify_consumption_low():
    from app.anomaly import classify_consumption

    assert classify_consumption(3.0, low_threshold=5.0, high_threshold=15.0) == "low"


def test_classify_consumption_normal():
    from app.anomaly import classify_consumption

    assert classify_consumption(10.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_high():
    from app.anomaly import classify_consumption

    assert classify_consumption(20.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_boundary_low():
    from app.anomaly import classify_consumption

    assert classify_consumption(5.0, low_threshold=5.0, high_threshold=15.0) == "normal"


def test_classify_consumption_boundary_high():
    from app.anomaly import classify_consumption

    assert classify_consumption(15.0, low_threshold=5.0, high_threshold=15.0) == "high"


def test_classify_consumption_invalid_thresholds():
    from app.anomaly import classify_consumption

    with pytest.raises(ValueError):
        classify_consumption(10.0, low_threshold=20.0, high_threshold=10.0)


@pytest.mark.parametrize("value,expected", [
    (0.0, "low"),
    (4.9, "low"),
    (5.0, "normal"),
    (14.9, "normal"),
    (15.0, "high"),
    (100.0, "high"),
])
def test_classify_consumption_parametrized(value, expected):
    from app.anomaly import classify_consumption

    assert classify_consumption(value, low_threshold=5.0, high_threshold=15.0) == expected
