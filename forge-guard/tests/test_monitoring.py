"""Tests for drift detection and prediction logging."""

from __future__ import annotations

import pytest

from app.monitoring import compute_drift, log_prediction


def test_compute_drift_no_drift():
    import numpy as np

    rng = np.random.default_rng(0)
    ref = list(rng.normal(50, 5, 100))
    cur = list(rng.normal(50, 5, 100))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert result["p_value"] > 0.0


def test_compute_drift_detects_shift():
    import numpy as np

    rng = np.random.default_rng(1)
    ref = list(rng.normal(50, 2, 200))
    cur = list(rng.normal(80, 2, 200))  # large mean shift
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [3.0])
    assert result["drift_detected"] is False
    assert result["p_value"] == 1.0


def test_log_prediction_persists_record(db_session):
    sensor = {
        "temperature": 78.0,
        "pressure": 50.0,
        "vibration": 2.0,
        "cycle_time": 30.0,
        "tool_wear": 10.0,
        "power_consumption": 100.0,
        "humidity": 45.0,
    }
    record = log_prediction(
        db=db_session,
        sensor_data=sensor,
        prediction=0,
        defect_probability=0.15,
        model_version="1.0.0",
        correlation_id="test-cid-001",
    )
    assert record.id is not None
    assert record.defect_probability == 0.15
    assert record.correlation_id == "test-cid-001"


def test_log_prediction_defect_case(db_session):
    sensor = {
        "temperature": 92.0,
        "pressure": 62.0,
        "vibration": 8.0,
        "cycle_time": 28.0,
        "tool_wear": 60.0,
        "power_consumption": 140.0,
        "humidity": 70.0,
    }
    record = log_prediction(
        db=db_session, sensor_data=sensor, prediction=1, defect_probability=0.88
    )
    assert record.prediction == 1
    assert record.defect_probability == 0.88


@pytest.mark.parametrize(
    "ref_mean,cur_mean,std,expect_drift",
    [
        (50, 51, 20, False),  # small shift relative to wide spread — no drift
        (50, 90, 1, True),  # large mean shift — clear drift
        (50, 50, 5, False),  # identical distributions — no drift
    ],
)
def test_drift_parametrized(ref_mean, cur_mean, std, expect_drift):
    import numpy as np

    rng = np.random.default_rng(42)
    ref = list(rng.normal(ref_mean, std, 300))
    cur = list(rng.normal(cur_mean, std, 300))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] == expect_drift


def test_defect_rate_empty_db(db_session):
    from app.monitoring import defect_rate

    result = defect_rate(db_session)
    assert result["total"] == 0
    assert result["defect_rate"] == 0.0


def test_defect_rate_with_mixed_predictions(db_session):
    from app.monitoring import defect_rate

    sensor = {
        "temperature": 75.0, "pressure": 50.0, "vibration": 2.0,
        "cycle_time": 30.0, "tool_wear": 10.0, "power_consumption": 100.0, "humidity": 45.0,
    }
    for i in range(4):
        log_prediction(db=db_session, sensor_data=sensor, prediction=0, defect_probability=0.1)
    for i in range(2):
        log_prediction(db=db_session, sensor_data=sensor, prediction=1, defect_probability=0.9)

    result = defect_rate(db_session)
    assert result["total"] == 6
    assert result["defects"] == 2
    assert abs(result["defect_rate"] - 1 / 3) < 0.01


def test_run_drift_check_empty_db_returns_empty(db_session):
    from app.monitoring import run_drift_check

    result = run_drift_check(db_session)
    assert result == []


def test_log_prediction_without_correlation_id(db_session):
    sensor = {
        "temperature": 75.0, "pressure": 50.0, "vibration": 2.0,
        "cycle_time": 30.0, "tool_wear": 10.0, "power_consumption": 100.0, "humidity": 45.0,
    }
    record = log_prediction(
        db=db_session, sensor_data=sensor, prediction=0, defect_probability=0.2
    )
    assert record.correlation_id is None
    assert record.model_version == "1.0.0"


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_log_prediction_various_probabilities(db_session, prob):
    sensor = {
        "temperature": 75.0, "pressure": 50.0, "vibration": 2.0,
        "cycle_time": 30.0, "tool_wear": 10.0, "power_consumption": 100.0, "humidity": 45.0,
    }
    record = log_prediction(
        db=db_session, sensor_data=sensor, prediction=int(prob > 0.5), defect_probability=prob
    )
    assert record.defect_probability == prob


def test_compute_zscore_outliers_no_outliers():
    from app.monitoring import compute_zscore_outliers

    values = [10.0] * 20  # identical values — no outliers
    result = compute_zscore_outliers(values)
    assert result["outlier_count"] == 0


def test_compute_zscore_outliers_detects_spike():
    from app.monitoring import compute_zscore_outliers

    values = [10.0] * 19 + [1000.0]  # one extreme outlier
    result = compute_zscore_outliers(values)
    assert result["outlier_count"] >= 1
    assert 19 in result["outlier_indices"]


def test_compute_zscore_outliers_insufficient_data():
    from app.monitoring import compute_zscore_outliers

    result = compute_zscore_outliers([1.0, 2.0])
    assert result["outlier_count"] == 0
    assert result["mean"] is None


def test_model_prediction_summary_empty(db_session):
    from app.monitoring import model_prediction_summary

    result = model_prediction_summary(db_session, "99.0.0")
    assert result["total"] == 0
    assert result["model_version"] == "99.0.0"


def test_model_prediction_summary_with_data(db_session):
    from app.monitoring import model_prediction_summary

    sensor = {
        "temperature": 75.0, "pressure": 50.0, "vibration": 2.0,
        "cycle_time": 30.0, "tool_wear": 10.0, "power_consumption": 100.0, "humidity": 45.0,
    }
    for _ in range(3):
        log_prediction(db=db_session, sensor_data=sensor, prediction=0, defect_probability=0.1,
                       model_version="test-v1")
    log_prediction(db=db_session, sensor_data=sensor, prediction=1, defect_probability=0.9,
                   model_version="test-v1")

    result = model_prediction_summary(db_session, "test-v1")
    assert result["total"] == 4
    assert result["defects"] == 1
    assert abs(result["defect_rate"] - 0.25) < 0.01
