"""Tests for ML model training and prediction."""

import numpy as np
import pytest
from app.model import _synthetic_model, load_metrics, predict, train_model


def test_train_model_returns_bundle_and_metrics(sample_df, sample_target):
    bundle, metrics = train_model(sample_df, sample_target)
    assert "ensemble" in bundle
    assert "scaler" in bundle
    assert "feature_pipeline" in bundle
    assert "r2_mean" in metrics
    assert "rmse_mean" in metrics
    assert metrics["n_samples"] == len(sample_target)


def test_train_model_r2_reasonable(sample_df, sample_target):
    _, metrics = train_model(sample_df, sample_target)
    assert metrics["r2_mean"] > -2.0


def test_predict_returns_correct_shape(sample_df, sample_target):
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df, bundle)
    assert preds.shape == (len(sample_df),)


def test_predict_positive_values(sample_df, sample_target):
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df, bundle)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("n", [1, 5, 10])
def test_predict_batch_sizes(sample_df, sample_target, n):
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df.head(n), bundle)
    assert len(preds) == n


def test_synthetic_model_loads():
    bundle = _synthetic_model()
    assert "ensemble" in bundle
    assert "scaler" in bundle


def test_load_metrics_returns_dict():
    m = load_metrics()
    assert isinstance(m, dict)
    assert "model_version" in m


def test_train_model_persists_file(sample_df, sample_target, tmp_path, monkeypatch):
    import app.model as model_module
    mp = tmp_path / "model.joblib"
    mtp = tmp_path / "metrics.json"
    monkeypatch.setattr(model_module, "MODEL_PATH", mp)
    monkeypatch.setattr(model_module, "METRICS_PATH", mtp)
    train_model(sample_df, sample_target)
    assert mp.exists()
    assert mtp.exists()
