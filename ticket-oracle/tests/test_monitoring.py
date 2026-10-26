"""Drift detection and monitoring tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.monitoring import compute_ks_drift, compute_psi, log_prediction, run_drift_check


@pytest.fixture()
def reference_dist():
    import numpy as np
    rng = np.random.default_rng(0)
    return rng.normal(0.3, 0.1, 1000).tolist()


@pytest.fixture()
def stable_dist():
    import numpy as np
    rng = np.random.default_rng(1)
    return rng.normal(0.3, 0.1, 1000).tolist()


@pytest.fixture()
def drifted_dist():
    import numpy as np
    rng = np.random.default_rng(2)
    return rng.normal(0.8, 0.05, 1000).tolist()


def test_ks_drift_stable(reference_dist, stable_dist):
    result = compute_ks_drift(reference_dist, stable_dist)
    assert result["drift_detected"] is False


def test_ks_drift_detected(reference_dist, drifted_dist):
    result = compute_ks_drift(reference_dist, drifted_dist)
    assert result["drift_detected"] is True


def test_ks_drift_keys(reference_dist, stable_dist):
    result = compute_ks_drift(reference_dist, stable_dist)
    for key in ["ks_statistic", "p_value", "drift_detected", "sample_size"]:
        assert key in result


def test_ks_drift_sample_size(reference_dist, stable_dist):
    result = compute_ks_drift(reference_dist, stable_dist)
    assert result["sample_size"] == len(stable_dist)


def test_ks_drift_insufficient_data():
    result = compute_ks_drift([0.1, 0.2], [0.3, 0.4])
    assert result["drift_detected"] is False


def test_ks_statistic_in_range(reference_dist, drifted_dist):
    result = compute_ks_drift(reference_dist, drifted_dist)
    assert 0.0 <= result["ks_statistic"] <= 1.0


def test_psi_stable_near_zero(reference_dist, stable_dist):
    psi = compute_psi(reference_dist, stable_dist)
    assert psi < 0.1


def test_psi_drifted_high(reference_dist, drifted_dist):
    psi = compute_psi(reference_dist, drifted_dist)
    assert psi >= 0.1


def test_psi_non_negative(reference_dist, stable_dist):
    psi = compute_psi(reference_dist, stable_dist)
    assert psi >= 0.0


def test_psi_insufficient_data():
    psi = compute_psi([0.1], [0.2])
    assert psi == 0.0


def test_run_drift_check_persists(db_session, reference_dist, stable_dist):
    result = run_drift_check(db_session, "test_feature", reference_dist, stable_dist)
    assert "ks_statistic" in result
    assert "psi_value" in result
    assert result["feature_name"] == "test_feature"


def test_log_prediction_persists(db_session):
    prediction = {
        "priority": "P2",
        "priority_probabilities": {"P1": 0.1, "P2": 0.6, "P3": 0.2, "P4": 0.1},
        "sla_breach_risk": 0.72,
        "sla_breach_predicted": True,
        "estimated_resolution_hours": 6.0,
        "confidence": 0.6,
    }
    row = log_prediction(db_session, "T-12345", prediction)
    assert row.id is not None
    assert row.ticket_id == "T-12345"
    assert row.priority_predicted == "P2"
    assert row.sla_breach_predicted is True


def test_log_prediction_probabilities_stored(db_session):
    prediction = {
        "priority": "P1",
        "priority_probabilities": {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02},
        "sla_breach_risk": 0.90,
        "sla_breach_predicted": True,
        "estimated_resolution_hours": 1.5,
        "confidence": 0.85,
    }
    row = log_prediction(db_session, "T-00001", prediction)
    assert abs(row.priority_p1_prob - 0.85) < 0.001


@pytest.mark.parametrize("priority", ["P1", "P2", "P3", "P4"])
def test_log_prediction_all_priorities(db_session, priority):
    prediction = {
        "priority": priority,
        "priority_probabilities": {"P1": 0.25, "P2": 0.25, "P3": 0.25, "P4": 0.25},
        "sla_breach_risk": 0.5,
        "sla_breach_predicted": False,
        "estimated_resolution_hours": 10.0,
        "confidence": 0.25,
    }
    row = log_prediction(db_session, f"T-{priority}", prediction)
    assert row.priority_predicted == priority
