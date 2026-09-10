"""Drift detection and monitoring tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    LatencyTimer,
    compute_drift,
    set_reference_window,
)


def test_compute_drift_no_drift() -> None:
    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    cur = list(np.random.default_rng(2).normal(10, 2, 200))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert not result["drift_detected"]


def test_compute_drift_detects_shift() -> None:
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    cur = list(np.random.default_rng(2).normal(30, 1, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"], f"Expected drift, p={result['p_value']}"
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data() -> None:
    result = compute_drift([1.0, 2.0], [3.0, 4.0])
    assert not result["drift_detected"]
    assert result["reason"] == "insufficient_data"


def test_compute_drift_p_value_range() -> None:
    ref = list(range(100))
    cur = list(range(100, 200))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_set_reference_window() -> None:
    values = list(range(600))
    set_reference_window(values)
    from app.monitoring import _reference_window

    assert len(_reference_window) == 500


def test_latency_timer() -> None:
    import time

    with LatencyTimer() as t:
        time.sleep(0.01)
    assert t.ms >= 5.0


def test_log_prediction(db_session) -> None:
    from datetime import datetime

    from app.monitoring import log_prediction

    log_prediction(db_session, "bldg-test", datetime.utcnow(), 15.5, 12.3)
    from app.database import PredictionLog

    count = db_session.query(PredictionLog).filter(PredictionLog.building_id == "bldg-test").count()
    assert count >= 1


def test_log_anomaly(db_session) -> None:
    from datetime import datetime

    from app.monitoring import log_anomaly

    log_anomaly(db_session, "bldg-test", datetime.utcnow(), 99.9, -0.6, 1, "critical")
    from app.database import AnomalyLog

    count = db_session.query(AnomalyLog).filter(AnomalyLog.building_id == "bldg-test").count()
    assert count >= 1


@pytest.mark.parametrize("mean_shift", [0, 5, 10, 20])
def test_drift_various_shifts(mean_shift: float) -> None:
    rng = np.random.default_rng(42)
    ref = list(rng.normal(10, 2, 200))
    cur = list(rng.normal(10 + mean_shift, 2, 200))
    result = compute_drift(ref, cur)
    if mean_shift >= 10:
        assert result["drift_detected"], f"Expected drift at shift={mean_shift}"


def test_latency_timer_ms_positive() -> None:
    with LatencyTimer() as t:
        pass
    assert t.ms >= 0.0


def test_compute_drift_exact_same_distribution() -> None:
    data = list(np.random.default_rng(99).normal(5, 1, 100))
    result = compute_drift(data, data)
    assert not result["drift_detected"]
    assert result["ks_statistic"] == pytest.approx(0.0)


def test_set_reference_window_truncates_to_500() -> None:
    large = list(range(1000))
    set_reference_window(large)
    from app.monitoring import _reference_window

    assert len(_reference_window) == 500
    assert _reference_window[0] == 500


def test_get_anomaly_stats_empty_db(db_session) -> None:
    from app.monitoring import get_anomaly_stats

    stats = get_anomaly_stats(db_session)
    assert stats["total_anomalies"] == 0
    assert stats["anomaly_rate"] == 0.0


def test_get_anomaly_stats_with_data(db_session) -> None:
    from datetime import datetime

    from app.database import AnomalyLog
    from app.monitoring import get_anomaly_stats

    for i, is_anomaly in enumerate([1, 0, 1]):
        severity = "critical" if is_anomaly else "none"
        entry = AnomalyLog(
            building_id=f"bldg-{i}",
            timestamp=datetime.utcnow(),
            consumption_kwh=10.0 + i,
            anomaly_score=-0.3 - i * 0.1,
            is_anomaly=is_anomaly,
            severity=severity,
        )
        db_session.add(entry)
    db_session.commit()
    stats = get_anomaly_stats(db_session)
    assert stats["total_anomalies"] == 3
    assert stats["critical_count"] == 2


def test_compute_feature_drift_summary_no_reference() -> None:
    from app.monitoring import compute_feature_drift_summary, reset_reference_window

    reset_reference_window()
    result = compute_feature_drift_summary({"temp": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]})
    assert len(result) == 1
    assert result[0]["feature"] == "temp"


def test_compute_feature_drift_summary_with_reference() -> None:
    from app.monitoring import compute_feature_drift_summary

    ref = [10.0 + i * 0.1 for i in range(50)]
    current = {"temp": [20.0 + i * 0.1 for i in range(20)]}
    result = compute_feature_drift_summary(current, reference=ref)
    assert len(result) == 1
    assert "ks_statistic" in result[0]
    assert "p_value" in result[0]
    assert "drift_detected" in result[0]


def test_summarize_drift_history_empty() -> None:
    from app.monitoring import summarize_drift_history

    result = summarize_drift_history([])
    assert result["total_checks"] == 0
    assert result["drift_count"] == 0
    assert result["drift_rate"] == 0.0


def test_summarize_drift_history_no_drift() -> None:
    from app.monitoring import summarize_drift_history

    checks = [
        {"drift_detected": False, "ks_statistic": 0.05, "p_value": 0.8},
        {"drift_detected": False, "ks_statistic": 0.03, "p_value": 0.9},
    ]
    result = summarize_drift_history(checks)
    assert result["total_checks"] == 2
    assert result["drift_count"] == 0
    assert result["drift_rate"] == 0.0


def test_summarize_drift_history_all_drift() -> None:
    from app.monitoring import summarize_drift_history

    checks = [
        {"drift_detected": True, "ks_statistic": 0.4, "p_value": 0.01},
        {"drift_detected": True, "ks_statistic": 0.5, "p_value": 0.02},
    ]
    result = summarize_drift_history(checks)
    assert result["drift_count"] == 2
    assert result["drift_rate"] == pytest.approx(1.0)


def test_summarize_drift_history_keys() -> None:
    from app.monitoring import summarize_drift_history

    checks = [{"drift_detected": False, "ks_statistic": 0.1, "p_value": 0.3}]
    result = summarize_drift_history(checks)
    assert set(result.keys()) == {"total_checks", "drift_count", "drift_rate", "mean_ks_statistic", "min_p_value"}


@pytest.mark.parametrize("n_drift,n_total", [(0, 5), (2, 5), (5, 5)])
def test_summarize_drift_history_parametrized(n_drift, n_total) -> None:
    from app.monitoring import summarize_drift_history

    checks = [{"drift_detected": i < n_drift, "ks_statistic": 0.1, "p_value": 0.3} for i in range(n_total)]
    result = summarize_drift_history(checks)
    assert result["drift_count"] == n_drift
    assert result["total_checks"] == n_total


def test_get_reference_window_size_empty() -> None:
    from app.monitoring import get_reference_window_size, reset_reference_window

    reset_reference_window()
    assert get_reference_window_size() == 0


def test_get_reference_window_size_after_set() -> None:
    from app.monitoring import get_reference_window_size, set_reference_window

    set_reference_window(list(range(50)))
    assert get_reference_window_size() == 50


def test_is_reference_window_ready_false_when_empty() -> None:
    from app.monitoring import is_reference_window_ready, reset_reference_window

    reset_reference_window()
    assert not is_reference_window_ready(min_samples=10)


def test_is_reference_window_ready_true_when_sufficient() -> None:
    from app.monitoring import is_reference_window_ready, set_reference_window

    set_reference_window([5.0] * 20)
    assert is_reference_window_ready(min_samples=10)


@pytest.mark.parametrize(
    "n_samples,min_samples,expected",
    [
        (0, 10, False),
        (9, 10, False),
        (10, 10, True),
        (100, 10, True),
        (5, 5, True),
    ],
)
def test_is_reference_window_ready_parametrized(n_samples, min_samples, expected) -> None:
    from app.monitoring import is_reference_window_ready, set_reference_window

    set_reference_window([1.0] * n_samples)
    assert is_reference_window_ready(min_samples=min_samples) == expected


class TestReferenceWindowStats:
    def setup_method(self) -> None:
        from app.monitoring import reset_reference_window

        reset_reference_window()

    def test_empty_window(self) -> None:
        from app.monitoring import reference_window_stats

        result = reference_window_stats()
        assert result["size"] == 0
        assert result["mean"] is None

    def test_populated_window(self) -> None:
        from app.monitoring import reference_window_stats, set_reference_window

        set_reference_window([1.0, 2.0, 3.0, 4.0, 5.0])
        result = reference_window_stats()
        assert result["size"] == 5
        assert result["mean"] == pytest.approx(3.0)

    def test_stats_keys(self) -> None:
        from app.monitoring import reference_window_stats, set_reference_window

        set_reference_window([10.0, 20.0])
        s = reference_window_stats()
        for key in ("size", "mean", "min", "max", "std"):
            assert key in s

    def test_single_value(self) -> None:
        from app.monitoring import reference_window_stats, set_reference_window

        set_reference_window([42.0])
        s = reference_window_stats()
        assert s["min"] == s["max"] == pytest.approx(42.0)
        assert s["std"] == pytest.approx(0.0)


class TestZscoreAlert:
    def test_no_alerts(self) -> None:
        from app.monitoring import zscore_alert

        result = zscore_alert([1.0, 1.0, 1.0, 1.0, 1.0])
        assert result == []

    def test_detects_outlier(self) -> None:
        from app.monitoring import zscore_alert

        values = [1.0] * 19 + [100.0]
        result = zscore_alert(values, threshold=3.0)
        assert 19 in result

    def test_invalid_threshold(self) -> None:
        from app.monitoring import zscore_alert

        with pytest.raises(ValueError):
            zscore_alert([1.0, 2.0], threshold=0.0)

    def test_too_short_raises(self) -> None:
        from app.monitoring import zscore_alert

        with pytest.raises(ValueError):
            zscore_alert([5.0])


class TestDriftSeverity:
    def test_no_drift(self) -> None:
        from app.monitoring import drift_severity

        assert drift_severity(0.5) == "low"

    def test_moderate(self) -> None:
        from app.monitoring import drift_severity

        assert drift_severity(0.03) == "medium"

    def test_severe(self) -> None:
        from app.monitoring import drift_severity

        assert drift_severity(0.001) == "critical"

    def test_boundary(self) -> None:
        from app.monitoring import drift_severity

        assert drift_severity(0.05) == "low"


class TestRollingAnomalyRate:
    def test_all_normal(self) -> None:
        from app.monitoring import rolling_anomaly_rate

        result = rolling_anomaly_rate([False] * 5)
        assert all(r == 0.0 for r in result)

    def test_all_anomaly(self) -> None:
        from app.monitoring import rolling_anomaly_rate

        result = rolling_anomaly_rate([True] * 5)
        assert all(r == pytest.approx(1.0) for r in result)

    def test_length_preserved(self) -> None:
        from app.monitoring import rolling_anomaly_rate

        assert len(rolling_anomaly_rate([True, False, True, False], window=2)) == 3

    def test_empty(self) -> None:
        from app.monitoring import rolling_anomaly_rate

        assert rolling_anomaly_rate([]) == []

    def test_invalid_window(self) -> None:
        from app.monitoring import rolling_anomaly_rate

        with pytest.raises(ValueError):
            rolling_anomaly_rate([True], window=0)


class TestAlertCountByLevel:
    def test_basic(self) -> None:
        from app.monitoring import alert_count_by_level

        alerts = [{"level": "warn"}, {"level": "error"}, {"level": "warn"}]
        result = alert_count_by_level(alerts)
        assert result["warn"] == 2
        assert result["error"] == 1

    def test_empty(self) -> None:
        from app.monitoring import alert_count_by_level

        assert alert_count_by_level([]) == {}

    def test_missing_level_key(self) -> None:
        from app.monitoring import alert_count_by_level

        result = alert_count_by_level([{}])
        assert "unknown" in result


class TestPValueToConfidence:
    @pytest.mark.parametrize(
        "p_value,expected",
        [
            (0.0, 100.0),
            (1.0, 0.0),
            (0.05, 95.0),
            (0.5, 50.0),
        ],
    )
    def test_conversion(self, p_value: float, expected: float) -> None:
        from app.monitoring import p_value_to_confidence

        assert p_value_to_confidence(p_value) == pytest.approx(expected)

    def test_clamp_above_one(self) -> None:
        from app.monitoring import p_value_to_confidence

        assert p_value_to_confidence(1.5) == pytest.approx(0.0)

    def test_clamp_below_zero(self) -> None:
        from app.monitoring import p_value_to_confidence

        assert p_value_to_confidence(-0.1) == pytest.approx(100.0)


class TestAlertSuppressionWindow:
    def test_within_cooldown_suppressed(self) -> None:
        from app.monitoring import alert_suppression_window

        assert alert_suppression_window(1000.0, 1100.0, cooldown_seconds=300.0) is True

    def test_outside_cooldown_not_suppressed(self) -> None:
        from app.monitoring import alert_suppression_window

        assert alert_suppression_window(1000.0, 1400.0, cooldown_seconds=300.0) is False

    def test_exactly_at_boundary_not_suppressed(self) -> None:
        from app.monitoring import alert_suppression_window

        assert alert_suppression_window(1000.0, 1300.0, cooldown_seconds=300.0) is False

    def test_zero_cooldown_raises(self) -> None:
        from app.monitoring import alert_suppression_window

        with pytest.raises(ValueError):
            alert_suppression_window(1000.0, 1100.0, cooldown_seconds=0)


class TestDriftTrend:
    def test_stable_with_few_values(self) -> None:
        from app.monitoring import drift_trend

        assert drift_trend([0.1, 0.2]) == "stable"

    def test_worsening(self) -> None:
        from app.monitoring import drift_trend

        p_values = [0.8, 0.7, 0.6, 0.5, 0.2, 0.1]
        assert drift_trend(p_values) == "worsening"

    def test_improving(self) -> None:
        from app.monitoring import drift_trend

        p_values = [0.1, 0.2, 0.5, 0.7, 0.8, 0.9]
        assert drift_trend(p_values) == "improving"

    def test_stable_consistent(self) -> None:
        from app.monitoring import drift_trend

        p_values = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        assert drift_trend(p_values) == "stable"


@pytest.mark.parametrize(
    "p_value,expected",
    [
        (0.001, "critical"),
        (0.01, "high"),
        (0.03, "medium"),
        (0.10, "low"),
    ],
)
def test_drift_severity_parametrized(p_value: float, expected: str) -> None:
    from app.monitoring import drift_severity

    assert drift_severity(p_value) == expected


@pytest.mark.parametrize(
    "error_rate,expected",
    [
        (0.01, "low"),
        (0.10, "medium"),
        (0.50, "high"),
    ],
)
def test_degradation_severity_parametrized(error_rate: float, expected: str) -> None:
    from app.monitoring import degradation_severity

    assert degradation_severity(error_rate) == expected


@pytest.mark.parametrize("p_value", [0.0, 0.05, 0.5, 1.0])
def test_p_value_to_confidence_in_range(p_value: float) -> None:
    from app.monitoring import p_value_to_confidence

    result = p_value_to_confidence(p_value)
    assert 0.0 <= result <= 100.0


@pytest.mark.parametrize(
    "flags,window,expected_len",
    [
        ([True, False, True, False, True], 3, 3),
        ([True, True, True, True], 2, 3),
    ],
)
def test_rolling_anomaly_rate_length(flags: list, window: int, expected_len: int) -> None:
    from app.monitoring import rolling_anomaly_rate

    result = rolling_anomaly_rate(flags, window=window)
    assert len(result) == expected_len


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_rolling_anomaly_rate_values_in_range(window: int) -> None:
    """All values from rolling_anomaly_rate are in [0.0, 1.0]."""
    from app.monitoring import rolling_anomaly_rate

    flags = [True, False, True, True, False, False, True]
    result = rolling_anomaly_rate(flags, window=window)
    assert all(0.0 <= r <= 1.0 for r in result)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_rolling_anomaly_rate_all_true(n: int) -> None:
    """rolling_anomaly_rate returns 1.0 for every window when all flags are True."""
    from app.monitoring import rolling_anomaly_rate

    flags = [True] * n
    result = rolling_anomaly_rate(flags, window=3)
    assert all(r == 1.0 for r in result)
