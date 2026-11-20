"""Model training and inference tests for Ticket-Oracle."""

from __future__ import annotations

import json

import pytest

from app.features import make_synthetic_dataset, payload_to_dataframe
from app.model import (
    _estimate_resolution_hours,
    load_models,
    predict,
    read_champion_metrics,
    train_models,
)


@pytest.fixture(autouse=True)
def tmp_model_dir(tmp_path, monkeypatch):
    """Redirect MODEL_DIR to a temporary directory for all model tests."""
    import app.model as m

    monkeypatch.setattr(m, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(m, "PRIORITY_MODEL_PATH", tmp_path / "priority_model.joblib")
    monkeypatch.setattr(m, "SLA_MODEL_PATH", tmp_path / "sla_model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    return tmp_path


@pytest.fixture(scope="module")
def small_dataset():
    X, y_p, y_s = make_synthetic_dataset(n_samples=300, random_state=0)
    return X, y_p, y_s


def test_train_returns_metrics(small_dataset, tmp_model_dir):

    X, y_p, y_s = small_dataset
    metrics = train_models(X, y_p, y_s)
    assert "priority_auc_mean" in metrics
    assert "sla_auc_mean" in metrics
    assert "n_samples" in metrics


def test_train_saves_model_files(small_dataset, tmp_model_dir):
    import app.model as m

    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    assert m.PRIORITY_MODEL_PATH.exists()
    assert m.SLA_MODEL_PATH.exists()


def test_train_saves_metrics_json(small_dataset, tmp_model_dir):
    import app.model as m

    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    with open(m.METRICS_PATH) as f:
        saved = json.load(f)
    assert "priority_auc_mean" in saved


def test_train_auc_above_chance(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    metrics = train_models(X, y_p, y_s)
    assert metrics["priority_auc_mean"] > 0.5
    assert metrics["sla_auc_mean"] > 0.5


def test_load_models_raises_when_missing(tmp_model_dir):
    with pytest.raises(FileNotFoundError):
        load_models()


def test_load_models_returns_pipelines(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    pp, sp = load_models()
    assert pp is not None
    assert sp is not None


def test_predict_output_keys(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    pp, sp = load_models()
    row = payload_to_dataframe({
        "description": "VPN not working for remote team after update",
        "department": "Engineering",
        "incident_type": "Network",
        "channel": "email",
        "customer_tier": "Gold",
        "product_area": "Infrastructure",
        "open_tickets_count": 40,
        "agent_load": 7,
        "hour_of_day": 14,
        "day_of_week": 1,
    })
    result = predict(pp, sp, row)
    for key in ["priority", "priority_probabilities", "sla_breach_risk",
                "sla_breach_predicted", "estimated_resolution_hours", "confidence"]:
        assert key in result


def test_predict_probability_sums_to_one(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    pp, sp = load_models()
    row = payload_to_dataframe({
        "description": "Security breach on production server",
        "department": "IT",
        "incident_type": "Security",
        "channel": "phone",
        "customer_tier": "Gold",
        "product_area": "Security",
        "open_tickets_count": 60,
        "agent_load": 10,
        "hour_of_day": 3,
        "day_of_week": 0,
    })
    result = predict(pp, sp, row)
    assert abs(sum(result["priority_probabilities"].values()) - 1.0) < 0.01


def test_predict_sla_risk_in_range(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    pp, sp = load_models()
    row = payload_to_dataframe({"description": "Minor software glitch", "department": "Finance",
                                "incident_type": "Software", "channel": "portal",
                                "customer_tier": "Bronze", "product_area": "ERP"})
    result = predict(pp, sp, row)
    assert 0.0 <= result["sla_breach_risk"] <= 1.0


@pytest.mark.parametrize("priority_idx,expected_min", [(0, 0.5), (1, 2.0), (2, 10.0), (3, 40.0)])
def test_estimate_resolution_hours(priority_idx, expected_min):
    hours = _estimate_resolution_hours(priority_idx, 0.5)
    assert hours >= expected_min


def test_read_champion_metrics_empty_when_no_file(tmp_model_dir):
    result = read_champion_metrics()
    assert result == {}


def test_read_champion_metrics_returns_saved(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    train_models(X, y_p, y_s)
    m = read_champion_metrics()
    assert "priority_auc_mean" in m


def test_model_version_in_metrics(small_dataset, tmp_model_dir):
    X, y_p, y_s = small_dataset
    metrics = train_models(X, y_p, y_s, model_version="test-1.2.3")
    assert metrics["model_version"] == "test-1.2.3"
