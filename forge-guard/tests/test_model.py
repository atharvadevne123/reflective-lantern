"""Tests for model training, persistence, and inference."""

from __future__ import annotations

import pytest


def test_train_model_returns_pipeline_and_metrics(feature_arrays):
    from app.model import train_model

    X, y = feature_arrays
    pipe, metrics = train_model(X, y, cv_folds=3)

    assert pipe is not None
    assert "auc_cv_mean" in metrics
    assert 0.5 <= metrics["auc_cv_mean"] <= 1.0
    assert metrics["n_features"] == X.shape[1]
    assert metrics["n_samples"] == X.shape[0]


def test_model_persisted_to_disk(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    model_file = tmp_path / "test_model.joblib"
    metrics_file = tmp_path / "test_metrics.json"
    monkeypatch.setattr(m, "MODEL_PATH", model_file)
    monkeypatch.setattr(m, "METRICS_PATH", metrics_file)

    X, y = feature_arrays
    m.train_model(X, y, cv_folds=2)

    assert model_file.exists()
    assert metrics_file.exists()


def test_predict_returns_valid_label_and_prob(feature_arrays):
    from app.model import predict, train_model

    X, y = feature_arrays
    pipe, _ = train_model(X, y, cv_folds=2)

    for i in range(min(5, len(X))):
        label, prob = predict(pipe, X[i : i + 1])
        assert label in (0, 1)
        assert 0.0 <= prob <= 1.0


def test_load_model_trains_if_missing(tmp_path, monkeypatch):
    from app import model as m

    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "nonexistent.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
    loaded = m.load_model()
    assert loaded is not None


def test_get_metrics_returns_dict_after_training(feature_arrays, tmp_path, monkeypatch):
    from app import model as m

    metrics_file = tmp_path / "metrics.json"
    monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(m, "METRICS_PATH", metrics_file)

    X, y = feature_arrays
    _, metrics = m.train_model(X, y, cv_folds=2)
    result = m.get_metrics()

    assert isinstance(result, dict)
    assert result["auc_cv_mean"] == metrics["auc_cv_mean"]


@pytest.mark.parametrize("cv_folds", [2, 3])
def test_cv_folds_parameter(feature_arrays, cv_folds):
    from app.model import train_model

    X, y = feature_arrays
    _, metrics = train_model(X, y, cv_folds=cv_folds)
    assert metrics["cv_folds"] == cv_folds
