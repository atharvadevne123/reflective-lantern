"""Drift detection and prediction logging tests."""

from __future__ import annotations

import pytest

from app.monitoring import (
    build_current_window,
    compute_drift,
    log_prediction,
    set_reference_distribution,
)


def test_compute_drift_no_drift():
    import numpy as np

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000).tolist()
    current = rng.normal(0, 1, 1000).tolist()  # large samples → high p-value
    result = compute_drift(ref, current)
    assert "ks_statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    # With large, identically-distributed samples KS p-value should be well above 0.05
    assert result["p_value"] > 0.01


def test_compute_drift_detects_shift():
    import numpy as np

    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500).tolist()
    current = rng.normal(5, 1, 200).tolist()  # Large mean shift
    result = compute_drift(ref, current)
    assert result["drift_detected"] is True
    assert result["ks_statistic"] > 0.5


def test_compute_drift_insufficient_data():
    result = compute_drift([1.0, 2.0], [1.5])
    assert result["drift_detected"] is False
    assert result["p_value"] == 1.0


@pytest.mark.parametrize("ref_size,curr_size", [
    (100, 50), (500, 200), (1000, 100)
])
def test_compute_drift_various_sizes(ref_size, curr_size):
    import numpy as np

    rng = np.random.default_rng(7)
    ref = rng.normal(0, 1, ref_size).tolist()
    current = rng.normal(0, 1, curr_size).tolist()
    result = compute_drift(ref, current)
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_set_reference_distribution():
    set_reference_distribution("test_feature", [1.0, 2.0, 3.0])
    from app.monitoring import _reference_features
    assert "test_feature" in _reference_features
    assert _reference_features["test_feature"] == [1.0, 2.0, 3.0]


def test_log_prediction_persists(db_session):
    entry = log_prediction(
        db=db_session,
        ticket_id="test-123",
        subject="VPN issue",
        body="VPN drops every 10 minutes.",
        priority="high",
        predicted_category="network",
        sla_breach_prob=0.75,
        resolution_hours_pred=4.5,
        confidence=0.88,
    )
    assert entry.id is not None
    assert entry.ticket_id == "test-123"
    assert entry.predicted_category == "network"
    assert entry.sla_breach_prob == pytest.approx(0.75, abs=0.01)


def test_build_current_window_keys(db_session):
    log_prediction(
        db=db_session,
        ticket_id="w1",
        subject="Test",
        body="Test body",
        priority="low",
        predicted_category="software",
        sla_breach_prob=0.1,
        resolution_hours_pred=8.0,
        confidence=0.9,
    )
    from app.monitoring import get_recent_predictions
    recent = get_recent_predictions(db_session, hours=24)
    window = build_current_window(recent)
    assert "text_length" in window
    assert "sla_breach_prob" in window
    assert "resolution_hours_pred" in window
    assert "confidence" in window


def test_volume_anomaly_empty_returns_false():
    from app.monitoring import detect_volume_anomaly
    result = detect_volume_anomaly([])
    assert result["is_anomaly"] is False


def test_volume_anomaly_normal_rate():
    from app.monitoring import detect_volume_anomaly
    result = detect_volume_anomaly([], baseline_hourly_rate=10.0)
    assert result["is_anomaly"] is False
