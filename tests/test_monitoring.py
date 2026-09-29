"""Tests for drift detection and prediction logging."""
from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import compute_drift, log_prediction, seed_reference_buffer


def test_compute_drift_detects_drift():
    ref = list(np.random.default_rng(0).normal(0, 1, 200))
    cur = list(np.random.default_rng(1).normal(5, 1, 200))  # clearly shifted
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0


def test_compute_drift_no_drift():
    rng = np.random.default_rng(42)
    ref = list(rng.normal(0, 1, 300))
    cur = list(rng.normal(0, 1, 300))  # same distribution
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [3.0])
    assert result["drift_detected"] is False
    assert result["ks_statistic"] is None


def test_compute_drift_returns_expected_keys():
    ref = list(range(50))
    cur = list(range(100, 150))
    result = compute_drift(ref, cur)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result


def test_log_prediction_creates_record(db_session):
    pred = log_prediction(
        db_session,
        carrier="DHL",
        distance_km=42.5,
        weight_kg=3.2,
        route_type="urban",
        hour_of_day=14,
        day_of_week=2,
        predicted_minutes=87.3,
        confidence=0.92,
    )
    assert pred.id is not None
    assert pred.carrier == "DHL"
    assert pred.predicted_minutes == pytest.approx(87.3)


def test_log_prediction_persists_to_db(db_session):
    from app.database import Prediction

    before_count = db_session.query(Prediction).count()
    log_prediction(
        db_session,
        carrier="FedEx",
        distance_km=100.0,
        weight_kg=10.0,
        route_type="highway",
        hour_of_day=9,
        day_of_week=0,
        predicted_minutes=200.0,
        confidence=0.85,
    )
    after_count = db_session.query(Prediction).count()
    assert after_count == before_count + 1


def test_seed_reference_buffer():
    samples = [
        {"distance_km": 50.0, "weight_kg": 5.0, "predicted_minutes": 120.0}
        for _ in range(20)
    ]
    seed_reference_buffer(samples)


@pytest.mark.parametrize("shift", [0, 3, 10])
def test_drift_scales_with_shift(shift):
    ref = list(np.random.default_rng(0).normal(0, 1, 200))
    cur = list(np.random.default_rng(1).normal(shift, 1, 200))
    result = compute_drift(ref, cur)
    if shift == 0:
        assert result["drift_detected"] is False
    elif shift >= 10:
        assert result["drift_detected"] is True


def test_compute_drift_ks_statistic_range():
    ref = list(np.arange(100, dtype=float))
    cur = list(np.arange(200, 300, dtype=float))
    result = compute_drift(ref, cur)
    assert 0.0 < result["ks_statistic"] <= 1.0


def test_compute_drift_p_value_range():
    ref = list(np.random.default_rng(5).normal(0, 1, 100))
    cur = list(np.random.default_rng(6).normal(0, 1, 100))
    result = compute_drift(ref, cur)
    assert 0.0 <= result["p_value"] <= 1.0


def test_log_prediction_with_model_version(db_session):
    pred = log_prediction(
        db_session,
        carrier="UPS",
        distance_km=75.0,
        weight_kg=2.0,
        route_type="suburban",
        hour_of_day=10,
        day_of_week=3,
        predicted_minutes=95.0,
        confidence=0.88,
        model_version="2.0.0",
    )
    assert pred.model_version == "2.0.0"


def test_seed_reference_buffer_with_empty_list():
    """Seeding with an empty list should not raise."""
    seed_reference_buffer([])


def test_run_drift_check_returns_status_key(db_session):
    from app.monitoring import run_drift_check

    result = run_drift_check(db_session)
    assert "status" in result
