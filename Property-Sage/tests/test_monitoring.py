"""Drift detection and monitoring tests."""

import numpy as np
import pytest

from app.monitoring import compute_drift, get_prediction_stats, log_prediction


def test_compute_drift_no_drift():
    rng = np.random.default_rng(0)
    ref = list(rng.normal(100, 10, 200))
    cur = list(rng.normal(100, 10, 200))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False
    assert 0 <= result["ks_statistic"] <= 1
    assert 0 <= result["p_value"] <= 1


def test_compute_drift_detects_shift():
    rng = np.random.default_rng(1)
    ref = list(rng.normal(100, 10, 300))
    cur = list(rng.normal(200, 10, 300))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_too_few_samples():
    ref = [100.0] * 200
    cur = [200.0] * 5
    result = compute_drift(ref, cur)
    assert result["drift_detected"] is False
    assert result["p_value"] == 1.0


def test_compute_drift_returns_required_keys():
    result = compute_drift([1.0] * 50, [1.0] * 50)
    assert {"ks_statistic", "p_value", "drift_detected", "sample_size"} <= result.keys()


@pytest.mark.parametrize("loc_ref,loc_cur,expect_drift", [
    (100, 100, False),
    (100, 500, True),
    (100, 100, False),
])
def test_drift_parametrized(loc_ref, loc_cur, expect_drift):
    rng = np.random.default_rng(99)
    ref = list(rng.normal(loc_ref, 5, 300))
    cur = list(rng.normal(loc_cur, 5, 300))
    result = compute_drift(ref, cur)
    assert result["drift_detected"] == expect_drift


def test_log_prediction_persists(db_session):
    log_prediction(
        db=db_session,
        request_id="test-uuid-123",
        input_data={
            "bedrooms": 3, "bathrooms": 2.0, "sqft": 1500.0,
            "lot_size": 5000.0, "year_built": 2005,
            "neighborhood": "suburb", "property_type": "house",
        },
        output={"predicted_price": 450_000.0, "predicted_rental_yield": 0.055},
    )
    from app.database import Prediction
    record = db_session.query(Prediction).filter_by(request_id="test-uuid-123").first()
    assert record is not None
    assert record.predicted_price == 450_000.0


def test_get_prediction_stats_empty(db_session):
    stats = get_prediction_stats(db_session)
    assert "total_predictions" in stats
    assert "predictions_last_24h" in stats
