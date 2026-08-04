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
