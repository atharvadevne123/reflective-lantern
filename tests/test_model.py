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


def test_model_version_string_format():
    """MODEL_VERSION in main.py should follow SemVer X.Y.Z."""
    import re

    from app.main import MODEL_VERSION

    assert re.match(r"^\d+\.\d+\.\d+$", MODEL_VERSION), f"Bad version: {MODEL_VERSION}"


def test_predict_single_row(trained_model):
    """Predict on a single row should return a 1-element array."""
    from app.model import predict

    pipe, _, X, _ = trained_model
    single = X[:1]
    preds = predict(pipe, single)
    assert preds.shape == (1,)
    assert preds[0] > 0


def test_train_with_minimal_data():
    """Model should train on the minimum viable dataset without error."""
    from app.model import train_model

    df = generate_synthetic_data(n=20, seed=99)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    pipe, metrics = train_model(X, y)
    assert pipe is not None
    assert metrics["n_samples"] == 20


def test_metrics_n_samples_matches_input(trained_model):
    """Reported n_samples should equal the number of training rows."""
    pipe, metrics, X, y = trained_model
    assert metrics["n_samples"] == len(y)


def test_predict_all_positive_on_large_batch():
    """All predictions across a larger synthetic batch should be positive."""
    from app.model import predict, train_model

    df = generate_synthetic_data(n=800, seed=3)
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    pipe, _ = train_model(X, y)
    preds = predict(pipe, X)
    assert np.all(preds > 0), "Some predictions are non-positive"
