"""Drift detection and monitoring tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    LatencyTimer,
    compute_drift,
    set_reference_window,
)


def test_compute_drift_no_drift():
    ref = list(np.random.default_rng(1).normal(10, 2, 200))
    cur = list(np.random.default_rng(2).normal(10, 2, 200))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert not result["drift_detected"]


def test_compute_drift_detects_shift():
    ref = list(np.random.default_rng(1).normal(10, 1, 200))
    cur = list(np.random.default_rng(2).normal(30, 1, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"], f"Expected drift, p={result['p_value']}"
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [3.0, 4.0])
    assert not result["drift_detected"]
    assert result["reason"] == "insufficient_data"


def test_compute_drift_p_value_range():
    ref = list(range(100))
    cur = list(range(100, 200))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_set_reference_window():
    values = list(range(600))
    set_reference_window(values)
    from app.monitoring import _reference_window

    assert len(_reference_window) == 500


def test_latency_timer():
    import time

    with LatencyTimer() as t:
        time.sleep(0.01)
    assert t.ms >= 5.0


def test_log_prediction(db_session):
    from datetime import datetime

    from app.monitoring import log_prediction

    log_prediction(db_session, "bldg-test", datetime.utcnow(), 15.5, 12.3)
    from app.database import PredictionLog

    count = db_session.query(PredictionLog).filter(PredictionLog.building_id == "bldg-test").count()
    assert count >= 1


def test_log_anomaly(db_session):
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
