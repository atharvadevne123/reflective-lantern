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


def test_compute_drift_detects_shift() -> None:
    ref = [1.0] * 50
    current = [100.0] * 50
    result = compute_drift(ref, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_no_shift() -> None:
    import random

    rng = random.Random(42)
    ref = [rng.gauss(10, 1) for _ in range(100)]
    current = [rng.gauss(10, 1) for _ in range(100)]
    result = compute_drift(ref, current)
    assert result["p_value"] > 0.01


def test_compute_drift_insufficient_data() -> None:
    result = compute_drift([1.0] * 5, [2.0] * 50)
    assert result["drift_detected"] is False
    assert result["reason"] == "insufficient_data"


def test_set_reference_window_updates() -> None:
    set_reference_window([1.0, 2.0, 3.0] * 20)
    assert get_reference_window_size() == 60


def test_set_reference_window_caps_at_max() -> None:
    set_reference_window([1.0] * 1000)
    assert get_reference_window_size() == 500


def test_is_reference_window_ready_false() -> None:
    assert not is_reference_window_ready()


def test_is_reference_window_ready_true() -> None:
    set_reference_window([5.0] * 20)
    assert is_reference_window_ready(min_samples=10)


@pytest.mark.parametrize(
    "n_samples,min_samples,expected",
    [
        (5, 10, False),
        (10, 10, True),
        (20, 15, True),
    ],
)
def test_is_reference_window_ready_parametrized(n_samples, min_samples, expected) -> None:
    set_reference_window([1.0] * n_samples)
    assert is_reference_window_ready(min_samples=min_samples) == expected


def test_compute_feature_drift_summary_multiple_features() -> None:
    set_reference_window([1.0] * 100)
    results = compute_feature_drift_summary(
        {
            "temperature": [1.0] * 30,
            "humidity": [100.0] * 30,
        }
    )
    assert len(results) == 2
    features = {r["feature"] for r in results}
    assert "temperature" in features
    assert "humidity" in features


def test_summarize_drift_history_empty() -> None:
    summary = summarize_drift_history([])
    assert summary["total_checks"] == 0
    assert summary["drift_count"] == 0
    assert summary["drift_rate"] == 0.0


def test_summarize_drift_history_all_drifted() -> None:
    results = [
        {"drift_detected": True, "ks_statistic": 0.9, "p_value": 0.001},
        {"drift_detected": True, "ks_statistic": 0.8, "p_value": 0.002},
    ]
    summary = summarize_drift_history(results)
    assert summary["drift_count"] == 2
    assert summary["drift_rate"] == 1.0
    assert summary["min_p_value"] == 0.001


def test_summarize_drift_history_mixed() -> None:
    results = [
        {"drift_detected": True, "ks_statistic": 0.9, "p_value": 0.001},
        {"drift_detected": False, "ks_statistic": 0.1, "p_value": 0.8},
    ]
    summary = summarize_drift_history(results)
    assert summary["total_checks"] == 2
    assert summary["drift_count"] == 1
    assert summary["drift_rate"] == 0.5


def test_reference_window_stats_empty() -> None:
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([])
    stats = reference_window_stats()
    assert stats["size"] == 0
    assert stats["mean"] is None


def test_reference_window_stats_populated() -> None:
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = reference_window_stats()
    assert stats["size"] == 5
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["min"] == pytest.approx(10.0)
    assert stats["max"] == pytest.approx(50.0)
    assert stats["std"] is not None


def test_reference_window_stats_has_all_keys() -> None:
    from app.monitoring import reference_window_stats, set_reference_window

    set_reference_window([5.0, 10.0, 15.0])
    stats = reference_window_stats()
    for key in ("size", "mean", "min", "max", "std"):
        assert key in stats


def test_compute_drift_equal_distributions() -> None:
    values = [float(i) for i in range(50)]
    result = compute_drift(values, values)
    assert result["ks_statistic"] == pytest.approx(0.0)
    assert result["p_value"] >= 0.05


@pytest.mark.parametrize("shift", [10.0, 50.0, 100.0])
def test_compute_drift_detects_increasing_shift(shift) -> None:
    ref = [0.0] * 50
    current = [shift] * 50
    result = compute_drift(ref, current)
    assert result["drift_detected"] is True


def test_reset_reference_window_clears_data() -> None:
    set_reference_window([1.0] * 30)
    reset_reference_window()
    assert get_reference_window_size() == 0


def test_compute_feature_drift_summary_returns_list() -> None:
    set_reference_window([1.0] * 50)
    results = compute_feature_drift_summary({"feat": [1.0] * 20})
    assert isinstance(results, list)
    assert len(results) == 1


def test_summarize_drift_history_single_entry() -> None:
    results = [{"drift_detected": False, "ks_statistic": 0.05, "p_value": 0.6}]
    summary = summarize_drift_history(results)
    assert summary["total_checks"] == 1
    assert summary["drift_count"] == 0
    assert summary["drift_rate"] == 0.0


class TestAlertRate:
    def test_basic_window(self) -> None:
        from app.monitoring import alert_rate

        assert alert_rate([1, 2, 3, 4, 5], window=3) == pytest.approx(4.0, abs=0.01)

    def test_empty_returns_zero(self) -> None:
        from app.monitoring import alert_rate

        assert alert_rate([]) == 0.0

    def test_window_larger_than_list(self) -> None:
        from app.monitoring import alert_rate

        assert alert_rate([10, 20], window=10) == pytest.approx(15.0, abs=0.01)


class TestErrorBudgetRemaining:
    def test_budget_intact(self) -> None:
        from app.monitoring import error_budget_remaining

        result = error_budget_remaining(0.999, 1.0, period_minutes=10080)
        assert result > 0

    def test_budget_breached(self) -> None:
        from app.monitoring import error_budget_remaining

        result = error_budget_remaining(0.999, 0.98, period_minutes=10080)
        assert result < 0

    def test_exact_slo_met(self) -> None:
        from app.monitoring import error_budget_remaining

        result = error_budget_remaining(0.99, 0.99, period_minutes=10000)
        assert result == pytest.approx(0.0, abs=0.01)


class TestDegradationSeverity:
    def test_healthy(self) -> None:
        from app.monitoring import degradation_severity

        assert degradation_severity(0.005) == "healthy"

    def test_degraded(self) -> None:
        from app.monitoring import degradation_severity

        assert degradation_severity(0.03) == "low"

    def test_critical(self) -> None:
        from app.monitoring import degradation_severity

        assert degradation_severity(0.5) == "high"
