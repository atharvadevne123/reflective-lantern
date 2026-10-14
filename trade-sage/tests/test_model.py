"""Tests for ML model training and prediction."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from app.features import FEATURE_NAMES


@pytest.fixture()
def tmp_model_paths(tmp_path):
    model_path = str(tmp_path / "model.joblib")
    metrics_path = str(tmp_path / "metrics.json")
    os.environ["MODEL_PATH"] = model_path
    os.environ["METRICS_PATH"] = metrics_path
    yield model_path, metrics_path
    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("METRICS_PATH", None)


def _make_xy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    score = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * rng.standard_normal(n)
    y = (score > 0).astype(int)
    return X, y


def test_train_model_returns_pipeline(tmp_model_paths):
    from app.model import train_model
    X, y = _make_xy()
    pipe, metrics = train_model(X, y)
    assert hasattr(pipe, "predict_proba")
    assert "auc_mean" in metrics


def test_train_model_creates_files(tmp_model_paths):
    model_path, metrics_path = tmp_model_paths
    from app.model import train_model
    X, y = _make_xy()
    train_model(X, y)
    assert os.path.exists(model_path)
    assert os.path.exists(metrics_path)


def test_metrics_json_valid(tmp_model_paths):
    _, metrics_path = tmp_model_paths
    from app.model import train_model
    X, y = _make_xy()
    train_model(X, y)
    with open(metrics_path) as f:
        data = json.load(f)
    assert 0.0 <= data["auc_mean"] <= 1.0
    assert data["n_features"] == len(FEATURE_NAMES)


def test_predict_output_shape(tmp_model_paths):
    from app.model import predict, train_model
    X, y = _make_xy()
    pipe, _ = train_model(X, y)
    preds, confs = predict(pipe, X[:5])
    assert len(preds) == 5
    assert len(confs) == 5


def test_predict_confidence_in_range(tmp_model_paths):
    from app.model import predict, train_model
    X, y = _make_xy()
    pipe, _ = train_model(X, y)
    _, confs = predict(pipe, X)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_predict_classes_binary(tmp_model_paths):
    from app.model import predict, train_model
    X, y = _make_xy()
    pipe, _ = train_model(X, y)
    preds, _ = predict(pipe, X)
    assert all(p in (0, 1) for p in preds)


def test_load_model_roundtrip(tmp_model_paths):
    from app.model import load_model, predict, train_model
    X, y = _make_xy()
    pipe, _ = train_model(X, y)
    loaded = load_model()
    preds1, _ = predict(pipe, X[:10])
    preds2, _ = predict(loaded, X[:10])
    assert preds1 == preds2


def test_load_model_missing_raises(tmp_path):
    os.environ["MODEL_PATH"] = str(tmp_path / "nonexistent_model.joblib")
    os.environ["METRICS_PATH"] = str(tmp_path / "nonexistent_metrics.json")
    from app.model import load_model
    with pytest.raises(FileNotFoundError):
        load_model()


def test_read_champion_auc(tmp_model_paths):
    from app.model import read_champion_auc, train_model
    assert read_champion_auc() == 0.0
    X, y = _make_xy()
    _, metrics = train_model(X, y)
    auc = read_champion_auc()
    assert auc == metrics["auc_mean"]


@pytest.mark.parametrize("n_samples", [200, 500, 1000])
def test_train_various_sizes(tmp_model_paths, n_samples):
    from app.model import train_model
    X, y = _make_xy(n=n_samples)
    _, metrics = train_model(X, y)
    assert metrics["n_samples"] == n_samples
