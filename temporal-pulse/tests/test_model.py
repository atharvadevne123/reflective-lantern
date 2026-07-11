"""Tests for ML model training and prediction in Temporal-Pulse."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestAnomalyDetector:
    def test_train_returns_model_and_scaler(self, feature_matrix):
        from app.model import train_anomaly_detector
        df, feature_cols = feature_matrix
        X = df[feature_cols].to_numpy(dtype=np.float32)
        model, scaler = train_anomaly_detector(X, contamination=0.05)
        assert model is not None
        assert scaler is not None

    def test_trained_model_is_isolation_forest(self, feature_matrix):
        from sklearn.ensemble import IsolationForest

        from app.model import train_anomaly_detector
        df, feature_cols = feature_matrix
        X = df[feature_cols].to_numpy(dtype=np.float32)
        model, _ = train_anomaly_detector(X)
        assert isinstance(model, IsolationForest)

    def test_score_anomaly_returns_array(self, trained_models):
        from app.model import score_anomaly
        X = trained_models["X"]
        scores = score_anomaly(X, trained_models["if_model"], trained_models["scaler"])
        assert len(scores) == len(X)

    def test_scores_in_range_zero_to_one(self, trained_models):
        from app.model import score_anomaly
        X = trained_models["X"]
        scores = score_anomaly(X, trained_models["if_model"], trained_models["scaler"])
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_score_anomaly_without_training_raises(self):
        from app import model as m
        m._IF_MODEL = None
        m._SCALER = None
        with pytest.raises(RuntimeError):
            from app.model import score_anomaly
            score_anomaly(np.random.rand(5, 10).astype(np.float32))

    def test_score_single_row(self, trained_models):
        from app.model import score_anomaly
        X = trained_models["X"][:1]
        scores = score_anomaly(X, trained_models["if_model"], trained_models["scaler"])
        assert len(scores) == 1

    @pytest.mark.parametrize("contamination", [0.01, 0.05, 0.10])
    def test_contamination_parameter(self, feature_matrix, contamination):
        from app.model import train_anomaly_detector
        df, feature_cols = feature_matrix
        X = df[feature_cols].to_numpy(dtype=np.float32)
        model, scaler = train_anomaly_detector(X, contamination=contamination)
        assert model is not None


class TestForecaster:
    def test_train_forecaster_returns_metrics(self, trained_models):
        assert "mae" in trained_models["metrics"]
        assert "rmse" in trained_models["metrics"]

    def test_metrics_are_non_negative(self, trained_models):
        assert trained_models["metrics"]["mae"] >= 0.0
        assert trained_models["metrics"]["rmse"] >= 0.0

    def test_predict_forecast_returns_list(self, trained_models):
        from app.model import predict_forecast
        X = trained_models["X"]
        preds = predict_forecast(X, horizon=3, rf_model=trained_models["rf_model"])
        assert isinstance(preds, list)
        assert len(preds) == 3

    def test_predict_forecast_without_training_raises(self):
        from app import model as m
        m._RF_MODEL = None
        with pytest.raises(RuntimeError):
            from app.model import predict_forecast
            predict_forecast(np.random.rand(5, 10).astype(np.float32), horizon=2)

    def test_feature_importance_returns_list(self, trained_models):
        from app.model import get_feature_importance
        fi = get_feature_importance(trained_models["rf_model"], trained_models["feature_cols"])
        assert isinstance(fi, list)
        assert len(fi) <= 20

    def test_feature_importance_has_keys(self, trained_models):
        from app.model import get_feature_importance
        fi = get_feature_importance(trained_models["rf_model"], trained_models["feature_cols"])
        if fi:
            assert "feature" in fi[0]
            assert "importance" in fi[0]

    def test_feature_importance_sorted_descending(self, trained_models):
        from app.model import get_feature_importance
        fi = get_feature_importance(trained_models["rf_model"], trained_models["feature_cols"])
        if len(fi) > 1:
            importances = [item["importance"] for item in fi]
            assert importances == sorted(importances, reverse=True)

    @pytest.mark.parametrize("horizon", [1, 3, 5, 10])
    def test_forecast_multiple_horizons(self, trained_models, horizon):
        from app.model import predict_forecast
        X = trained_models["X"]
        preds = predict_forecast(X, horizon=horizon, rf_model=trained_models["rf_model"])
        assert len(preds) == horizon
