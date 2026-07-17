"""Extended tests for monitoring utilities."""

from __future__ import annotations

import pytest

from app.monitoring import (
    compute_drift,
    compute_feature_drift_summary,
    get_reference_window_size,
    is_reference_window_ready,
    reset_reference_window,
    set_reference_window,
    summarize_drift_history,
)


@pytest.fixture(autouse=True)
def reset_window():
    """Reset the reference window before and after each test."""
    reset_reference_window()
    yield
    reset_reference_window()


def test_compute_drift_detects_shift():
    ref = [1.0] * 50
    current = [100.0] * 50
    result = compute_drift(ref, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_no_shift():
    import random

    rng = random.Random(42)
    ref = [rng.gauss(10, 1) for _ in range(100)]
    current = [rng.gauss(10, 1) for _ in range(100)]
    result = compute_drift(ref, current)
    assert result["p_value"] > 0.01


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0] * 5, [2.0] * 50)
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_data"


def test_set_reference_window_updates():
    set_reference_window([1.0, 2.0, 3.0] * 20)
    assert get_reference_window_size() == 60


def test_set_reference_window_caps_at_max():
    set_reference_window([1.0] * 1000)
    assert get_reference_window_size() == 500


def test_is_reference_window_ready_false():
    assert not is_reference_window_ready()


def test_is_reference_window_ready_true():
    set_reference_window([5.0] * 20)
    assert is_reference_window_ready(min_samples=10)


@pytest.mark.parametrize("n_samples,min_samples,expected", [
    (5, 10, False),
    (10, 10, True),
    (20, 15, True),
])
def test_is_reference_window_ready_parametrized(n_samples, min_samples, expected):
    set_reference_window([1.0] * n_samples)
    assert is_reference_window_ready(min_samples=min_samples) == expected


def test_compute_feature_drift_summary_multiple_features():
    set_reference_window([1.0] * 100)
    results = compute_feature_drift_summary({
        "temperature": [1.0] * 30,
        "humidity": [100.0] * 30,
    })
    assert len(results) == 2
    features = {r["feature"] for r in results}
    assert "temperature" in features
    assert "humidity" in features


def test_summarize_drift_history_empty():
    summary = summarize_drift_history([])
    assert summary["total_checks"] == 0
    assert summary["drift_count"] == 0
    assert summary["drift_rate"] == 0.0


def test_summarize_drift_history_all_drifted():
    results = [
        {"drift_detected": True, "ks_statistic": 0.9, "p_value": 0.001},
        {"drift_detected": True, "ks_statistic": 0.8, "p_value": 0.002},
    ]
    summary = summarize_drift_history(results)
    assert summary["drift_count"] == 2
    assert summary["drift_rate"] == 1.0
    assert summary["min_p_value"] == 0.001


def test_summarize_drift_history_mixed():
    results = [
        {"drift_detected": True, "ks_statistic": 0.9, "p_value": 0.001},
        {"drift_detected": False, "ks_statistic": 0.1, "p_value": 0.8},
    ]
    summary = summarize_drift_history(results)
    assert summary["total_checks"] == 2
    assert summary["drift_count"] == 1
    assert summary["drift_rate"] == 0.5


def test_reference_window_stats_empty():
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([])
    stats = reference_window_stats()
    assert stats["size"] == 0
    assert stats["mean"] is None


def test_reference_window_stats_populated():
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = reference_window_stats()
    assert stats["size"] == 5
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(50.0)
    assert stats["std"] is not None


def test_reference_window_stats_has_all_keys():
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([5.0, 10.0, 15.0])
    stats = reference_window_stats()
    for key in ("size", "mean", "min", "max", "std"):
        assert key in stats
