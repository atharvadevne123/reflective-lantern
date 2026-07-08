"""Tests for model training and prediction logic."""

from __future__ import annotations

import pytest

from app.features import FEATURE_COLUMNS
from app.model import (
    CONGESTION_LABELS,
    generate_synthetic_data,
    predict,
    train_model,
)


def test_synthetic_data_shape() -> None:
    X, y = generate_synthetic_data(n_samples=200)
    assert len(X) == 200
    assert len(y) == 200


def test_synthetic_data_has_all_feature_columns() -> None:
    X, _ = generate_synthetic_data(n_samples=100)
    assert set(FEATURE_COLUMNS).issubset(set(X.columns))


def test_synthetic_labels_in_valid_range() -> None:
    _, y = generate_synthetic_data(n_samples=500)
    assert all(0 <= label <= 3 for label in y)


def test_synthetic_data_has_multiple_classes() -> None:
    _, y = generate_synthetic_data(n_samples=500)
    assert len(set(y)) > 1


def test_train_model_returns_two_pipelines() -> None:
    X, y = generate_synthetic_data(n_samples=300)
    models, metrics = train_model(X, y)
    assert set(models.keys()) == {"xgb", "lgbm"}


def test_train_model_metrics_structure() -> None:
    X, y = generate_synthetic_data(n_samples=300)
    _, metrics = train_model(X, y)
    assert "ensemble_auc" in metrics
    assert "cv_results" in metrics
    assert metrics["n_features"] == len(FEATURE_COLUMNS)


def test_train_model_auc_above_chance() -> None:
    X, y = generate_synthetic_data(n_samples=300)
    _, metrics = train_model(X, y)
    assert metrics["ensemble_auc"] > 0.55


@pytest.mark.parametrize("n_samples", [100, 500])
def test_train_model_various_sizes(n_samples: int) -> None:
    X, y = generate_synthetic_data(n_samples=n_samples)
    models, metrics = train_model(X, y)
    assert len(models) == 2
    assert metrics["n_samples"] == n_samples


def test_predict_output_keys(mock_models: dict) -> None:
    X, _ = generate_synthetic_data(n_samples=3)
    result = predict(mock_models, X.iloc[:1])
    expected = {
        "congestion_level",
        "congestion_label",
        "congestion_probability",
        "class_probabilities",
        "incident_score",
        "model_version",
    }
    assert expected.issubset(result.keys())


def test_predict_congestion_label_valid(mock_models: dict) -> None:
    X, _ = generate_synthetic_data(n_samples=3)
    result = predict(mock_models, X.iloc[:1])
    assert result["congestion_label"] in CONGESTION_LABELS.values()


def test_congestion_labels_complete() -> None:
    assert set(CONGESTION_LABELS.keys()) == {0, 1, 2, 3}
    assert "free" in CONGESTION_LABELS.values()
    assert "severe" in CONGESTION_LABELS.values()
