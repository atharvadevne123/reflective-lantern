"""Tests for drift detection and prediction health monitoring."""

from __future__ import annotations

import numpy as np
import pytest


def test_compute_drift_no_reference() -> None:
    from app.monitoring import compute_drift

    result = compute_drift("unknown_feature", np.array([1.0, 2.0, 3.0]))
    assert result["drift_detected"] is False
    assert "error" in result


def test_compute_drift_same_distribution() -> None:
    from app.monitoring import compute_drift, set_reference_distribution

    ref = np.random.default_rng(0).standard_normal(500)
    set_reference_distribution("f0", ref)
    result = compute_drift("f0", ref + np.random.default_rng(1).normal(0, 0.001, 500))
    assert "ks_statistic" in result
    assert result["p_value"] >= 0.0


def test_compute_drift_different_distribution() -> None:
    from app.monitoring import compute_drift, set_reference_distribution

    ref = np.zeros(200)
    set_reference_distribution("f1", ref)
    current = np.ones(200) * 100
    result = compute_drift("f1", current)
    assert result["drift_detected"] is True


def test_compute_drift_insufficient_samples() -> None:
    from app.monitoring import compute_drift, set_reference_distribution

    set_reference_distribution("f2", np.array([1.0, 2.0, 3.0]))
    result = compute_drift("f2", np.array([1.0]))
    assert result["drift_detected"] is False


def test_prediction_health_empty() -> None:
    from app.monitoring import prediction_health

    health = prediction_health()
    assert health["total_predictions"] == 0


def test_log_prediction_increments_count() -> None:
    from app.monitoring import log_prediction, prediction_health

    log_prediction({"label": "normal", "confidence": 0.9}, [0.0] * 25)
    health = prediction_health()
    assert health["total_predictions"] == 1


def test_prediction_health_attack_rate() -> None:
    from app.monitoring import log_prediction, prediction_health

    log_prediction({"label": "attack", "confidence": 0.8}, [1.0] * 25)
    log_prediction({"label": "normal", "confidence": 0.95}, [0.0] * 25)
    health = prediction_health()
    assert health["attack_rate"] == pytest.approx(0.5, abs=0.01)


def test_check_alerts_returns_messages() -> None:
    from app.monitoring import check_alerts

    drift_results = [
        {"feature_name": "f0", "drift_detected": True, "ks_statistic": 0.9, "p_value": 0.001},
    ]
    alerts = check_alerts(drift_results)
    assert len(alerts) == 1
    assert "DRIFT ALERT" in alerts[0]


def test_check_alerts_no_drift() -> None:
    from app.monitoring import check_alerts

    drift_results = [
        {"feature_name": "f0", "drift_detected": False, "ks_statistic": 0.01, "p_value": 0.8},
    ]
    alerts = check_alerts(drift_results)
    assert len(alerts) == 0


def test_reset_monitoring_clears_state() -> None:
    from app.monitoring import log_prediction, prediction_health, reset_monitoring

    log_prediction({"label": "attack", "confidence": 0.9}, [0.0] * 25)
    reset_monitoring()
    health = prediction_health()
    assert health["total_predictions"] == 0


@pytest.mark.parametrize("n_samples", [10, 50, 200])
def test_compute_drift_various_sizes(n_samples: int) -> None:
    from app.monitoring import compute_drift, reset_monitoring, set_reference_distribution

    reset_monitoring()
    ref = np.random.default_rng(99).standard_normal(500)
    set_reference_distribution("fx", ref)
    current = np.random.default_rng(n_samples).standard_normal(n_samples)
    result = compute_drift("fx", current)
    assert "ks_statistic" in result
    assert "p_value" in result


def test_get_feature_drift_summary_no_predictions() -> None:
    from app.monitoring import get_feature_drift_summary

    result = get_feature_drift_summary()
    assert "error" in result


def test_drift_threshold_default() -> None:
    from app.monitoring import DRIFT_THRESHOLD, get_drift_threshold

    assert get_drift_threshold() == DRIFT_THRESHOLD


def test_drift_threshold_env_override(monkeypatch) -> None:
    from app.monitoring import get_drift_threshold

    monkeypatch.setenv("DRIFT_THRESHOLD", "0.01")
    assert get_drift_threshold() == 0.01


def test_drift_threshold_invalid_env_falls_back(monkeypatch) -> None:
    from app.monitoring import DRIFT_THRESHOLD, get_drift_threshold

    monkeypatch.setenv("DRIFT_THRESHOLD", "not-a-number")
    assert get_drift_threshold() == DRIFT_THRESHOLD
