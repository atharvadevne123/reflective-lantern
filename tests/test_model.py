"""Tests for model training and prediction."""

from __future__ import annotations

import pytest


class TestModelTraining:
    def test_train_returns_model_and_metrics(self, synthetic_data):
        from app.model import train_model
        X, y = synthetic_data
        model, metrics = train_model(X.values[:200], y.values[:200], cv_folds=2)
        assert model is not None
        assert "r2_mean" in metrics
        assert "rmse_mean" in metrics

    def test_metrics_r2_positive(self, synthetic_data):
        from app.model import train_model
        X, y = synthetic_data
        _, metrics = train_model(X.values[:200], y.values[:200], cv_folds=2)
        assert metrics["r2_mean"] > 0.0

    def test_metrics_has_all_keys(self, synthetic_data):
        from app.model import train_model
        X, y = synthetic_data
        _, metrics = train_model(X.values[:200], y.values[:200], cv_folds=2)
        for key in ["r2_mean", "r2_std", "rmse_mean", "rmse_std", "n_samples", "n_features"]:
            assert key in metrics

    def test_model_file_created(self, tmp_path, synthetic_data, monkeypatch):
        from app import model as model_mod
        model_path = tmp_path / "test_model.joblib"
        metrics_path = tmp_path / "test_metrics.json"
        monkeypatch.setattr(model_mod, "MODEL_PATH", model_path)
        monkeypatch.setattr(model_mod, "METRICS_PATH", metrics_path)
        X, y = synthetic_data
        model_mod.train_model(X.values[:100], y.values[:100], cv_folds=2)
        assert model_path.exists()
        assert metrics_path.exists()

    def test_model_load_returns_none_if_missing(self, tmp_path, monkeypatch):
        from app import model as model_mod
        monkeypatch.setattr(model_mod, "MODEL_PATH", tmp_path / "nonexistent.joblib")
        assert model_mod.load_model() is None


class TestModelPrediction:
    def test_predict_returns_array(self, synthetic_data):
        from app.model import predict, train_model
        X, y = synthetic_data
        model, _ = train_model(X.values[:200], y.values[:200], cv_folds=2)
        preds = predict(model, X.values[:10])
        assert len(preds) == 10

    def test_predict_positive_values(self, synthetic_data):
        from app.model import predict, train_model
        X, y = synthetic_data
        model, _ = train_model(X.values[:200], y.values[:200], cv_folds=2)
        preds = predict(model, X.values[:20])
        assert all(p > 0 for p in preds)

    @pytest.mark.parametrize("n_rows", [1, 5, 50])
    def test_predict_various_batch_sizes(self, synthetic_data, n_rows):
        from app.model import predict, train_model
        X, y = synthetic_data
        model, _ = train_model(X.values[:200], y.values[:200], cv_folds=2)
        preds = predict(model, X.values[:n_rows])
        assert len(preds) == n_rows


class TestSyntheticData:
    def test_synthetic_data_shape(self, synthetic_data):
        X, y = synthetic_data
        assert X.shape[0] == 300
        assert X.shape[1] > 10

    def test_synthetic_loads_in_range(self, synthetic_data):
        _, y = synthetic_data
        assert y.min() >= 1000
        assert y.max() <= 8000

    def test_synthetic_data_no_nulls(self, synthetic_data):
        X, y = synthetic_data
        assert not X.isnull().any().any()
        assert not y.isnull().any()

    def test_custom_sample_size(self):
        from app.model import generate_synthetic_training_data
        X, y = generate_synthetic_training_data(n_samples=100)
        assert len(X) == 100
        assert len(y) == 100
