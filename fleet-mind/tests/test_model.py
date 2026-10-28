"""Tests for the Fleet-Mind ML model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.model import (
    MODEL_VERSION,
    _generate_training_data,
    get_metrics,
    predict,
    train_model,
)


class TestTrainModel:
    """Tests for model training."""

    def test_train_returns_metrics(self):
        metrics = train_model()
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics

    def test_roc_auc_above_baseline(self):
        metrics = train_model()
        assert metrics["roc_auc"] >= 0.7, "AUC should exceed random baseline"

    def test_cv_auc_reported(self):
        metrics = train_model()
        assert "cv_auc_mean" in metrics
        assert "cv_auc_std" in metrics
        assert metrics["cv_auc_std"] >= 0

    def test_metrics_all_between_zero_and_one(self):
        metrics = train_model()
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range"

    def test_train_samples_positive(self):
        metrics = train_model()
        assert metrics.get("train_samples", 0) > 0


class TestGenerateTrainingData:
    """Tests for synthetic data generation."""

    def test_shape(self):
        X, y = _generate_training_data(200)
        assert X.shape[0] == 200
        assert len(y) == 200

    def test_binary_labels(self):
        _, y = _generate_training_data(200)
        assert set(y).issubset({0, 1})

    def test_class_balance(self):
        _, y = _generate_training_data(1000)
        positive_rate = y.mean()
        assert 0.1 <= positive_rate <= 0.9


class TestPredict:
    """Tests for the predict() function."""

    @pytest.fixture(scope="class")
    def loaded_model(self):
        from app.model import load_model
        return load_model()

    def test_predict_returns_dict(self, loaded_model, sample_reading):
        from app.features import build_feature_pipeline
        import pandas as pd
        pipeline = build_feature_pipeline()
        raw = {k: v for k, v in sample_reading.items() if k != "vehicle_id"}
        features = pipeline.fit_transform(pd.DataFrame([raw]))
        result = predict(loaded_model, features, raw)
        assert isinstance(result, dict)

    def test_predict_keys(self, loaded_model, sample_reading):
        from app.features import build_feature_pipeline
        import pandas as pd
        pipeline = build_feature_pipeline()
        raw = {k: v for k, v in sample_reading.items() if k != "vehicle_id"}
        features = pipeline.fit_transform(pd.DataFrame([raw]))
        result = predict(loaded_model, features, raw)
        for key in ("maintenance_needed", "anomaly_score", "confidence", "model_version"):
            assert key in result

    def test_anomaly_score_in_range(self, loaded_model, sample_reading):
        from app.features import build_feature_pipeline
        import pandas as pd
        pipeline = build_feature_pipeline()
        raw = {k: v for k, v in sample_reading.items() if k != "vehicle_id"}
        features = pipeline.fit_transform(pd.DataFrame([raw]))
        result = predict(loaded_model, features, raw)
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_model_version_string(self, loaded_model, sample_reading):
        from app.features import build_feature_pipeline
        import pandas as pd
        pipeline = build_feature_pipeline()
        raw = {k: v for k, v in sample_reading.items() if k != "vehicle_id"}
        features = pipeline.fit_transform(pd.DataFrame([raw]))
        result = predict(loaded_model, features, raw)
        assert result["model_version"] == MODEL_VERSION

    @pytest.fixture
    def sample_reading(self):
        return {
            "speed_kmh": 80.0,
            "engine_temp_c": 90.0,
            "oil_pressure_bar": 3.5,
            "fuel_consumption_lph": 8.0,
            "battery_voltage_v": 12.8,
            "brake_pad_mm": 7.0,
            "tire_pressure_bar": 2.3,
            "rpm": 2500.0,
            "coolant_temp_c": 85.0,
            "transmission_temp_c": 75.0,
        }


class TestGetMetrics:
    """Tests for get_metrics()."""

    def test_returns_dict(self):
        train_model()
        assert isinstance(get_metrics(), dict)
