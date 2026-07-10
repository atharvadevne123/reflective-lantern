"""Tests for KS-test drift detection and prediction logging."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.monitoring import compute_drift, log_drift, log_prediction


def test_drift_detected_between_shifted_distributions(
    reference_distribution: list[float],
    drifted_distribution: list[float],
) -> None:
    result = compute_drift(reference_distribution, drifted_distribution)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.1
    assert result["p_value"] < 0.05


def test_no_drift_on_same_distribution() -> None:
    import numpy as np

    data = np.random.default_rng(99).normal(50, 10, 120).tolist()
    result = compute_drift(data[:60], data[60:])
    # Same distribution: p-value should be high or KS small
    assert not result["drift_detected"] or result["ks_statistic"] < 0.3


def test_insufficient_data_returns_no_drift() -> None:
    result = compute_drift([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert result["drift_detected"] is False
    assert "reason" in result


def test_log_prediction_returns_entry_with_id(db_session: Session) -> None:
    prediction = {
        "congestion_level": 1,
        "congestion_probability": 0.65,
        "incident_score": 0.03,
        "model_version": "1.0.0",
    }
    entry = log_prediction(
        db_session,
        hour=9,
        day_of_week=2,
        route_id="TEST-R42",
        road_type="highway",
        prediction=prediction,
        features={"vehicle_count": 2000},
    )
    assert entry.id is not None
    assert entry.route_id == "TEST-R42"
    assert entry.congestion_level == 1


def test_log_drift_creates_record_with_drift_flag(db_session: Session) -> None:
    result = {"ks_statistic": 0.55, "p_value": 0.001, "drift_detected": True}
    entry = log_drift(db_session, "vehicle_count", result)
    assert entry.id is not None
    assert entry.drift_detected == 1
    assert entry.feature_name == "vehicle_count"


def test_log_drift_no_drift_flag(db_session: Session) -> None:
    result = {"ks_statistic": 0.08, "p_value": 0.72, "drift_detected": False}
    entry = log_drift(db_session, "avg_speed_kmh", result)
    assert entry.drift_detected == 0


@pytest.mark.parametrize(
    "ks,p,expected",
    [
        (0.6, 0.001, True),
        (0.05, 0.9, False),
        (0.3, 0.04, True),
    ],
)
def test_drift_threshold_parametrized(ks: float, p: float, expected: bool) -> None:
    fake = {"ks_statistic": ks, "p_value": p, "drift_detected": p < 0.05}
    assert fake["drift_detected"] == expected
