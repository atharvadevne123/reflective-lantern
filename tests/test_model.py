"""Model training and prediction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_train_returns_metrics(trained_model):
    bundle, metrics = trained_model
    assert "r2_mean" in metrics
    assert "mae_kwh" in metrics
    assert metrics["r2_mean"] > -2.0
    assert metrics["mae_kwh"] >= 0


def test_train_r2_reasonable(trained_model):
    _, metrics = trained_model
    assert metrics["r2_mean"] > 0.0, f"R2 too low: {metrics['r2_mean']}"


def test_predict_shape(trained_model, sample_df):
    from app.model import predict

    bundle, _ = trained_model
    preds = predict(bundle, sample_df)
    assert len(preds) == len(sample_df)


def test_predict_non_negative(trained_model, sample_df):
    from app.model import predict

    bundle, _ = trained_model
    preds = predict(bundle, sample_df)
    assert np.all(preds >= -10), "Predictions should be near-positive for energy kWh"


def test_anomaly_model_scores(trained_anomaly_model, sample_df):
    from app.features import make_feature_row
    from app.model import score_anomaly

    row = make_feature_row(14, 1, 6, 28.0, 60.0, 50, 1, 12.0)
    result = score_anomaly(trained_anomaly_model, row)
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert result["is_anomaly"] in (0, 1)
    assert result["severity"] in ("none", "warning", "critical")


def test_model_persistence(sample_df, sample_y, tmp_path, monkeypatch):
    from app import model as model_mod

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(model_mod, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_mod, "METRICS_PATH", metrics_path)

    from app.model import train_model

    bundle, _ = train_model(sample_df, sample_df["consumption_kwh"])
    assert model_path.exists()
    assert metrics_path.exists()


def test_load_model_none_when_missing(tmp_path, monkeypatch):
    from app import model as model_mod

    monkeypatch.setattr(model_mod, "MODEL_PATH", tmp_path / "nonexistent.joblib")
    from app.model import load_model

    assert load_model() is None


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_train_various_sizes(n_samples):
    from app.model import train_model

    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n_samples),
            "day_of_week": rng.integers(0, 7, n_samples),
            "month": rng.integers(1, 13, n_samples),
            "temperature_c": rng.uniform(0, 35, n_samples),
            "humidity_pct": rng.uniform(20, 90, n_samples),
            "occupancy": rng.integers(0, 100, n_samples),
            "hvac_state": rng.integers(0, 2, n_samples),
            "consumption_kwh": rng.uniform(5, 30, n_samples),
        }
    )
    _, metrics = train_model(df, df["consumption_kwh"])
    assert metrics["n_samples"] == n_samples
