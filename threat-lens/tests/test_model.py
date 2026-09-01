"""Tests for model training and prediction."""

import pytest

from app.features import generate_synthetic_dataset
from app.model import ATTACK_CLASSES, build_pipeline, load_metrics, predict, train_model


def test_train_returns_pipeline_and_metrics() -> None:
    X, y = generate_synthetic_dataset(n_samples=300, seed=1)
    pipe, metrics = train_model(X, y)
    assert pipe is not None
    assert "accuracy_mean" in metrics
    assert 0.0 < metrics["accuracy_mean"] <= 1.0


def test_model_accuracy_above_threshold() -> None:
    X, y = generate_synthetic_dataset(n_samples=500, seed=2)
    _, metrics = train_model(X, y)
    # Synthetic dataset is well-separated; expect >70% accuracy
    assert metrics["accuracy_mean"] > 0.70


def test_predict_output_structure(trained_model: tuple) -> None:
    pipe, _ = trained_model
    X, _ = generate_synthetic_dataset(n_samples=10, seed=99)
    result = predict(pipe, X[:1])
    assert "predicted_class" in result
    assert "is_attack" in result
    assert "confidence" in result
    assert "class_probabilities" in result
    assert result["predicted_class"] in ATTACK_CLASSES


def test_predict_probabilities_sum_to_one(trained_model: tuple) -> None:
    pipe, _ = trained_model
    X, _ = generate_synthetic_dataset(n_samples=5, seed=7)
    for i in range(len(X)):
        result = predict(pipe, X[i : i + 1])
        prob_sum = sum(result["class_probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.02


def test_attack_classes_complete() -> None:
    assert set(ATTACK_CLASSES) == {"normal", "dos", "probe", "r2l", "u2r"}


def test_build_pipeline_has_scaler_and_ensemble() -> None:
    pipe = build_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "ensemble" in step_names


def test_metrics_file_written(trained_model: tuple) -> None:
    metrics = load_metrics()
    assert isinstance(metrics, dict)


@pytest.mark.parametrize("n_samples", [100, 200, 500])
def test_train_various_sizes(n_samples: int) -> None:
    X, y = generate_synthetic_dataset(n_samples=n_samples, seed=42)
    pipe, metrics = train_model(X, y)
    assert metrics["n_samples"] == n_samples
    assert pipe is not None
