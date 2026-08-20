"""Tests for model training and inference."""
from __future__ import annotations

import numpy as np
import pytest

from app.features import build_feature_pipeline, generate_synthetic_data, prepare_X


@pytest.fixture(scope="module")
def trained_model():
    from app.model import train_model

    df = generate_synthetic_data(n=400, seed=1)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    pipe, metrics = train_model(X, y)
    return pipe, metrics, X, y


def test_model_trains_without_error(trained_model):
    pipe, metrics, X, y = trained_model
    assert pipe is not None


def test_metrics_have_expected_keys(trained_model):
    _, metrics, _, _ = trained_model
    assert "rmse_mean" in metrics
    assert "r2_mean" in metrics
    assert "n_features" in metrics


def test_rmse_is_positive(trained_model):
    _, metrics, _, _ = trained_model
    assert metrics["rmse_mean"] > 0


def test_r2_is_reasonable(trained_model):
    _, metrics, _, _ = trained_model
    # Should explain at least 50% of variance on synthetic data
    assert metrics["r2_mean"] > 0.5


def test_predict_output_shape(trained_model):
    pipe, _, X, _ = trained_model
    from app.model import predict

    preds = predict(pipe, X)
    assert preds.shape == (X.shape[0],)


def test_predict_positive_values(trained_model):
    pipe, _, X, _ = trained_model
    from app.model import predict

    preds = predict(pipe, X)
    assert np.all(preds > 0)


def test_n_features_matches(trained_model):
    _, metrics, X, _ = trained_model
    assert metrics["n_features"] == X.shape[1]


@pytest.mark.parametrize("n_samples", [100, 500])
def test_model_scales_with_data(n_samples):
    from app.model import train_model

    df = generate_synthetic_data(n=n_samples, seed=7)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    pipe, metrics = train_model(X, y)
    assert metrics["n_samples"] == n_samples
