"""Tests for the automated retraining pipeline."""

import pytest

from pipelines.retrain_dag import (
    get_retrain_history,
    run_retrain_pipeline,
    should_retrain,
)


def test_should_retrain_with_drift():
    drift = [{"drift_detected": True, "p_value": 0.01}]
    assert should_retrain(drift) is True


def test_should_retrain_no_drift():
    drift = [{"drift_detected": False, "p_value": 0.8}]
    assert should_retrain(drift) is False


def test_should_retrain_empty():
    assert should_retrain([]) is False


def test_should_retrain_mixed():
    drift = [
        {"drift_detected": False, "p_value": 0.6},
        {"drift_detected": True, "p_value": 0.02},
    ]
    assert should_retrain(drift) is True


def test_run_retrain_skips_when_no_drift():
    no_drift = [{"drift_detected": False, "p_value": 0.9}]
    result = run_retrain_pipeline(n_samples=100, drift_results=no_drift)
    assert result["status"] == "skipped"
    assert result["reason"] == "no_drift"


def test_run_retrain_completes_with_drift(tmp_path, monkeypatch):
    import pipelines.retrain_dag as dag
    monkeypatch.setattr(dag, "RETRAIN_LOG", tmp_path / "retrain_log.json")
    import app.model as m
    monkeypatch.setattr(m, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(m, "PRICE_MODEL_PATH", tmp_path / "price_model.joblib")
    monkeypatch.setattr(m, "RENTAL_MODEL_PATH", tmp_path / "rental_model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")

    drift = [{"drift_detected": True, "p_value": 0.01}]
    result = run_retrain_pipeline(n_samples=200, drift_results=drift)
    assert result["status"] == "completed"
    assert "metrics" in result
    assert result["triggered_by_drift"] is True


def test_get_retrain_history_empty(tmp_path, monkeypatch):
    import pipelines.retrain_dag as dag
    monkeypatch.setattr(dag, "RETRAIN_LOG", tmp_path / "nope.json")
    assert get_retrain_history() == []


@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_retrain_accepts_various_sample_sizes(tmp_path, monkeypatch, n_samples):
    import app.model as m
    import pipelines.retrain_dag as dag
    monkeypatch.setattr(dag, "RETRAIN_LOG", tmp_path / "log.json")
    monkeypatch.setattr(m, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(m, "PRICE_MODEL_PATH", tmp_path / "price_model.joblib")
    monkeypatch.setattr(m, "RENTAL_MODEL_PATH", tmp_path / "rental_model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")

    result = run_retrain_pipeline(n_samples=n_samples, drift_results=[{"drift_detected": True, "p_value": 0.01}])
    assert result["n_samples"] == n_samples
