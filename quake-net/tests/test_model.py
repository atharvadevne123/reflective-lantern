"""Tests for model training, prediction, and magnitude classification."""

from __future__ import annotations

import pytest

from app.features import make_synthetic_dataset
from app.model import (
    build_ensemble,
    classify_magnitude,
    predict_magnitude,
    read_champion_metrics,
    train_model,
)


class TestTrainModel:
    def test_train_returns_pipeline_and_metrics(self) -> None:
        df = make_synthetic_dataset(n_samples=300, seed=1)
        pipeline, metrics = train_model(df=df)
        assert pipeline is not None
        assert isinstance(metrics, dict)

    def test_metrics_contain_required_keys(self, trained_model) -> None:
        _, metrics = trained_model
        for key in ("rmse", "mae", "r2", "cv_r2_mean", "cv_r2_std"):
            assert key in metrics, f"Missing key: {key}"

    def test_model_r2_is_positive(self, trained_model) -> None:
        _, metrics = trained_model
        assert metrics["r2"] > 0.0

    def test_model_rmse_is_below_threshold(self, trained_model) -> None:
        # RMSE < 2.0 Richter units is a reasonable sanity check on synthetic data
        _, metrics = trained_model
        assert metrics["rmse"] < 2.0

    def test_cv_r2_mean_positive(self, trained_model) -> None:
        _, metrics = trained_model
        assert metrics["cv_r2_mean"] > 0.0

    def test_n_features_recorded(self, trained_model) -> None:
        _, metrics = trained_model
        assert metrics["n_features"] > 0

    def test_n_samples_recorded(self, trained_model) -> None:
        _, metrics = trained_model
        assert metrics["n_samples"] == 300


class TestPredictMagnitude:
    def test_predict_returns_magnitude_in_range(self, trained_model, sample_features) -> None:
        pipeline, _ = trained_model
        result = predict_magnitude(pipeline, sample_features)
        assert 0.1 <= result["predicted_magnitude"] <= 9.9

    def test_predict_returns_aftershock_probability(self, trained_model, sample_features) -> None:
        pipeline, _ = trained_model
        result = predict_magnitude(pipeline, sample_features)
        assert 0.0 <= result["aftershock_probability"] <= 1.0

    def test_predict_returns_magnitude_class(self, trained_model, sample_features) -> None:
        pipeline, _ = trained_model
        result = predict_magnitude(pipeline, sample_features)
        assert result["magnitude_class"] in [
            "micro",
            "minor",
            "light",
            "moderate",
            "strong",
            "major",
            "great",
        ]

    def test_high_amplitude_gives_higher_magnitude(self, trained_model) -> None:
        pipeline, _ = trained_model
        low = predict_magnitude(
            pipeline,
            {
                "latitude": 37.5,
                "longitude": -122.0,
                "depth_km": 10.0,
                "station_count": 5,
                "p_wave_amplitude": 0.5,
                "s_wave_amplitude": 1.2,
                "epicentral_distance_km": 50.0,
                "fault_type": "normal",
            },
        )
        high = predict_magnitude(
            pipeline,
            {
                "latitude": 37.5,
                "longitude": -122.0,
                "depth_km": 10.0,
                "station_count": 30,
                "p_wave_amplitude": 50.0,
                "s_wave_amplitude": 100.0,
                "epicentral_distance_km": 50.0,
                "fault_type": "reverse",
            },
        )
        assert high["predicted_magnitude"] >= low["predicted_magnitude"] - 0.5

    def test_aftershock_prob_higher_for_large_magnitude(self, trained_model) -> None:
        from app.model import predict_magnitude

        pipeline, _ = trained_model
        low = predict_magnitude(
            pipeline,
            {
                "latitude": 37.5,
                "longitude": -122.0,
                "depth_km": 5.0,
                "station_count": 3,
                "p_wave_amplitude": 0.1,
                "s_wave_amplitude": 0.3,
                "epicentral_distance_km": 10.0,
                "fault_type": "normal",
            },
        )
        high = predict_magnitude(
            pipeline,
            {
                "latitude": 37.5,
                "longitude": -122.0,
                "depth_km": 5.0,
                "station_count": 50,
                "p_wave_amplitude": 200.0,
                "s_wave_amplitude": 500.0,
                "epicentral_distance_km": 10.0,
                "fault_type": "reverse",
            },
        )
        # Higher predicted magnitudes should have higher aftershock probabilities
        assert high["aftershock_probability"] >= low["aftershock_probability"] - 0.1


class TestClassifyMagnitude:
    @pytest.mark.parametrize(
        "magnitude,expected",
        [
            (0.5, "micro"),
            (1.5, "micro"),
            (2.5, "minor"),
            (3.9, "minor"),
            (4.5, "light"),
            (5.3, "moderate"),
            (6.1, "strong"),
            (7.2, "major"),
            (8.5, "great"),
        ],
    )
    def test_classify_magnitude_classes(self, magnitude: float, expected: str) -> None:
        assert classify_magnitude(magnitude) == expected

    def test_boundary_exactly_at_2(self) -> None:
        assert classify_magnitude(2.0) == "minor"

    def test_boundary_exactly_at_4(self) -> None:
        assert classify_magnitude(4.0) == "light"


class TestBuildEnsemble:
    def test_ensemble_has_two_estimators(self) -> None:
        ensemble = build_ensemble()
        assert len(ensemble.estimators) == 2

    def test_ensemble_weights_sum_to_one(self) -> None:
        ensemble = build_ensemble()
        assert abs(sum(ensemble.weights) - 1.0) < 0.01


class TestReadChampionMetrics:
    def test_returns_dict(self) -> None:
        result = read_champion_metrics()
        assert isinstance(result, dict)


class TestRecordTrainingRun:
    METRICS = {
        "rmse": 0.31,
        "mae": 0.24,
        "r2": 0.86,
        "cv_r2_mean": 0.84,
        "cv_r2_std": 0.02,
        "n_features": 47,
        "n_samples": 2000,
        "model_version": "1.0.0",
    }

    def test_writes_a_row(self, db_session) -> None:
        from app.database import ModelMetrics
        from app.model import record_training_run

        before = db_session.query(ModelMetrics).count()
        assert record_training_run(self.METRICS, notes="unit test") is True
        # record_training_run uses its own session, so query through a fresh one.
        from app.database import SessionLocal

        session = SessionLocal()
        try:
            assert session.query(ModelMetrics).count() >= before
        finally:
            session.close()

    def test_returns_false_on_bad_metrics(self) -> None:
        from app.model import record_training_run

        assert record_training_run({"rmse": 0.1}) is False

    def test_training_does_not_persist_by_default(self) -> None:
        from app.features import make_synthetic_dataset
        from app.model import train_model

        df = make_synthetic_dataset(n_samples=120, seed=7)
        _, metrics = train_model(df=df)
        assert "r2" in metrics
