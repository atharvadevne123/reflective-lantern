"""Tests for Ops-Vision ML model training and prediction."""

import numpy as np
import pytest
from sklearn.ensemble import VotingClassifier


class TestGenerateSyntheticData:
    """Tests for the synthetic data generator."""

    def test_returns_dataframe_and_series(self):
        """generate_synthetic_data returns (DataFrame, Series)."""
        from app.model import generate_synthetic_data

        df, labels = generate_synthetic_data(n_samples=100)
        assert len(df) == 100
        assert len(labels) == 100

    def test_incident_rate_approximately_correct(self):
        """Generated data has approximately the requested incident rate."""
        from app.model import generate_synthetic_data

        _, labels = generate_synthetic_data(n_samples=1000, incident_rate=0.3)
        actual_rate = labels.mean()
        assert abs(actual_rate - 0.3) < 0.05

    @pytest.mark.parametrize("n_samples", [50, 200, 500])
    def test_various_sample_sizes(self, n_samples):
        """Synthetic data generator works for multiple sample sizes."""
        from app.model import generate_synthetic_data

        df, labels = generate_synthetic_data(n_samples=n_samples)
        assert len(df) == n_samples

    def test_feature_columns_present(self):
        """All six metric columns appear in the generated DataFrame."""
        from app.model import generate_synthetic_data

        df, _ = generate_synthetic_data(n_samples=100)
        expected = [
            "cpu_usage_pct",
            "memory_usage_pct",
            "error_rate_per_min",
            "latency_p99_ms",
            "request_rate_per_sec",
            "disk_io_util_pct",
        ]
        for col in expected:
            assert col in df.columns

    def test_cpu_usage_in_valid_range(self):
        """CPU usage values must be in [0, 100]."""
        from app.model import generate_synthetic_data

        df, _ = generate_synthetic_data(n_samples=500)
        assert df["cpu_usage_pct"].between(0, 100).all()


class TestBuildModel:
    """Tests for build_model ensemble construction."""

    def test_build_model_returns_voting_classifier(self):
        """build_model() returns a VotingClassifier."""
        from app.model import build_model

        model = build_model()
        assert isinstance(model, VotingClassifier)

    def test_voting_is_soft(self):
        """Ensemble uses soft voting."""
        from app.model import build_model

        model = build_model()
        assert model.voting == "soft"

    def test_ensemble_has_rf(self):
        """Ensemble includes a RandomForest estimator."""
        from app.model import build_model

        model = build_model()
        names = [name for name, _ in model.estimators]
        assert "rf" in names


class TestTrainAndEvaluate:
    """Tests for the train() and evaluate() functions."""

    def test_train_returns_model_and_metrics(self, transformed_X, synthetic_labels):
        """train() returns (VotingClassifier, dict) tuple."""
        from app.model import train

        model, metrics = train(transformed_X, synthetic_labels, cv_folds=2)
        assert isinstance(model, VotingClassifier)
        assert "cv_auc_mean" in metrics
        assert "cv_auc_std" in metrics

    def test_cv_auc_in_valid_range(self, transformed_X, synthetic_labels):
        """Cross-validation AUC-ROC must be in [0, 1]."""
        from app.model import train

        _, metrics = train(transformed_X, synthetic_labels, cv_folds=2)
        assert 0.0 <= metrics["cv_auc_mean"] <= 1.0

    def test_evaluate_returns_test_auc(self, trained_model, transformed_X, synthetic_labels):
        """evaluate() returns test_auc_roc key."""
        from app.model import evaluate

        metrics = evaluate(trained_model, transformed_X, synthetic_labels)
        assert "test_auc_roc" in metrics
        assert 0.0 <= metrics["test_auc_roc"] <= 1.0

    @pytest.mark.parametrize("cv_folds", [2, 3])
    def test_train_with_different_cv_folds(self, transformed_X, synthetic_labels, cv_folds):
        """train() should work with different numbers of CV folds."""
        from app.model import train

        model, metrics = train(transformed_X, synthetic_labels, cv_folds=cv_folds)
        assert isinstance(model, VotingClassifier)


class TestPredict:
    """Tests for the predict() function."""

    def test_predict_returns_predictions_and_proba(self, trained_model, transformed_X):
        """predict() returns (preds array, proba array)."""
        from app.model import predict

        preds, proba = predict(trained_model, transformed_X)
        assert len(preds) == len(transformed_X)
        assert len(proba) == len(transformed_X)

    def test_predictions_are_binary(self, trained_model, transformed_X):
        """Predictions should be 0 or 1."""
        from app.model import predict

        preds, _ = predict(trained_model, transformed_X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_probabilities_in_range(self, trained_model, transformed_X):
        """All probabilities must be in [0, 1]."""
        from app.model import predict

        _, proba = predict(trained_model, transformed_X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


class TestModelPersistence:
    """Tests for save_model() and load_model()."""

    def test_save_and_load_roundtrip(self, trained_model, tmp_path):
        """Saved model can be loaded and produces identical predictions."""
        from app.model import load_model, save_model

        model_path = tmp_path / "test_model.pkl"
        save_model(trained_model, model_path)
        loaded = load_model(model_path)
        assert isinstance(loaded, VotingClassifier)

    def test_load_raises_if_file_missing(self, tmp_path):
        """load_model raises FileNotFoundError for missing path."""
        from app.model import load_model

        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")
