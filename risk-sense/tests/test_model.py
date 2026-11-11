"""Model training and prediction tests for Risk-Sense."""

from __future__ import annotations

import numpy as np
import pytest

from app.model import generate_synthetic_data, predict, train
from app.features import build_pipeline


class TestGenerateSyntheticData:
    def test_returns_correct_shape(self) -> None:
        X, y = generate_synthetic_data(n_samples=100)
        assert X.shape == (100, 7)
        assert y.shape == (100,)

    def test_binary_labels(self) -> None:
        _, y = generate_synthetic_data(n_samples=200)
        assert set(y).issubset({0, 1})

    def test_seed_reproducible(self) -> None:
        X1, y1 = generate_synthetic_data(n_samples=50, seed=99)
        X2, y2 = generate_synthetic_data(n_samples=50, seed=99)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_income_positive(self) -> None:
        X, _ = generate_synthetic_data(n_samples=100)
        assert (X[:, 6] > 0).all()


class TestModelTraining:
    def test_train_returns_model_and_metrics(self) -> None:
        X, y = generate_synthetic_data(n_samples=300, seed=1)
        pipeline = build_pipeline()
        model, metrics = train(X, y, pipeline)
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_contain_auc(self) -> None:
        X, y = generate_synthetic_data(n_samples=300, seed=2)
        pipeline = build_pipeline()
        _, metrics = train(X, y, pipeline)
        assert "cv_auc_mean" in metrics
        assert "train_auc" in metrics

    def test_cv_auc_above_chance(self) -> None:
        X, y = generate_synthetic_data(n_samples=500, seed=3)
        pipeline = build_pipeline()
        _, metrics = train(X, y, pipeline)
        assert metrics["cv_auc_mean"] > 0.5

    def test_train_records_sample_count(self) -> None:
        X, y = generate_synthetic_data(n_samples=200, seed=4)
        pipeline = build_pipeline()
        _, metrics = train(X, y, pipeline)
        assert metrics["n_samples"] == 200


class TestModelPrediction:
    def test_predict_returns_probabilities(self, trained_model_and_pipeline) -> None:
        model, pipeline, _ = trained_model_and_pipeline
        X, _ = generate_synthetic_data(n_samples=10, seed=10)
        proba = predict(model, X, pipeline)
        assert proba.shape == (10,)
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_predict_single_row(self, trained_model_and_pipeline) -> None:
        model, pipeline, _ = trained_model_and_pipeline
        X, _ = generate_synthetic_data(n_samples=1, seed=11)
        proba = predict(model, X, pipeline)
        assert len(proba) == 1

    def test_high_risk_features_score_higher(self, trained_model_and_pipeline) -> None:
        model, pipeline, _ = trained_model_and_pipeline
        low_risk = np.array([[0.1, 0.95, 0.05, 120, 3, 0, 100_000.0]])
        high_risk = np.array([[0.8, 0.05, 0.95, 6, 10, 5, 15_000.0]])
        low_score = predict(model, low_risk, pipeline)[0]
        high_score = predict(model, high_risk, pipeline)[0]
        assert high_score > low_score, f"Expected high_risk > low_risk: {high_score} vs {low_score}"

    @pytest.mark.parametrize("n_samples", [1, 5, 50, 100])
    def test_predict_various_batch_sizes(self, trained_model_and_pipeline, n_samples: int) -> None:
        model, pipeline, _ = trained_model_and_pipeline
        X, _ = generate_synthetic_data(n_samples=n_samples, seed=20)
        proba = predict(model, X, pipeline)
        assert len(proba) == n_samples
