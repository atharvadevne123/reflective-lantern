"""Tests for model training, prediction, and persistence."""

from __future__ import annotations

import pytest


class TestGenerateSyntheticData:
    def test_generates_correct_shape(self):
        from app.model import generate_synthetic_data

        X, y = generate_synthetic_data(500)
        assert len(X) == 500
        assert len(y) == 500

    def test_consumption_non_negative(self):
        from app.model import generate_synthetic_data

        _, y = generate_synthetic_data(200)
        assert (y >= 0).all()

    def test_has_expected_columns(self):
        from app.model import generate_synthetic_data

        X, _ = generate_synthetic_data(100)
        required = {
            "consumption_kwh",
            "temperature_c",
            "humidity_pct",
            "hour_of_day",
            "day_of_week",
        }
        assert required.issubset(set(X.columns))


class TestTrainModel:
    def test_train_returns_bundle_and_metrics(self):
        from app.model import generate_synthetic_data, train_model

        X, y = generate_synthetic_data(300)
        bundle, metrics = train_model(X, y)
        assert "pipeline" in bundle
        assert "model" in bundle
        assert "r2_mean" in metrics
        assert "rmse_mean" in metrics

    def test_r2_positive(self):
        from app.model import generate_synthetic_data, train_model

        X, y = generate_synthetic_data(300)
        _, metrics = train_model(X, y)
        assert metrics["r2_mean"] > 0.0

    def test_metrics_file_written(self, tmp_path, monkeypatch):
        import app.model as m

        monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
        from app.model import generate_synthetic_data, train_model

        X, y = generate_synthetic_data(200)
        train_model(X, y)
        assert (tmp_path / "metrics.json").exists()

    def test_model_file_written(self, tmp_path, monkeypatch):
        import app.model as m

        monkeypatch.setattr(m, "MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(m, "METRICS_PATH", tmp_path / "metrics.json")
        from app.model import generate_synthetic_data, train_model

        X, y = generate_synthetic_data(200)
        train_model(X, y)
        assert (tmp_path / "model.joblib").exists()


class TestPredict:
    def test_predict_returns_list(self):
        from app.model import generate_synthetic_data, predict, train_model

        X, y = generate_synthetic_data(300)
        bundle, _ = train_model(X, y)
        readings = X.head(5).to_dict(orient="records")
        preds = predict(bundle, readings)
        assert isinstance(preds, list)
        assert len(preds) == 5

    def test_predict_non_negative(self):
        from app.model import generate_synthetic_data, predict, train_model

        X, y = generate_synthetic_data(200)
        bundle, _ = train_model(X, y)
        preds = predict(bundle, X.head(10).to_dict(orient="records"))
        assert all(p >= 0 for p in preds)

    def test_predict_horizon_scales_output(self):
        from app.model import generate_synthetic_data, predict, train_model

        X, y = generate_synthetic_data(200)
        bundle, _ = train_model(X, y)
        reading = X.head(1).to_dict(orient="records")
        p1 = predict(bundle, reading, horizon_h=1)[0]
        p6 = predict(bundle, reading, horizon_h=6)[0]
        assert p6 >= p1

    @pytest.mark.parametrize("n_readings", [1, 3, 10])
    def test_predict_batch_sizes(self, n_readings):
        from app.model import generate_synthetic_data, predict, train_model

        X, y = generate_synthetic_data(200)
        bundle, _ = train_model(X, y)
        preds = predict(bundle, X.head(n_readings).to_dict(orient="records"))
        assert len(preds) == n_readings


class TestBuildEnsemble:
    def test_ensemble_has_three_estimators(self):
        from app.model import build_ensemble

        ens = build_ensemble()
        assert len(ens.estimators) == 3

    def test_ensemble_estimator_names(self):
        from app.model import build_ensemble

        ens = build_ensemble()
        names = [name for name, _ in ens.estimators]
        assert set(names) == {"xgb", "lgbm", "rf"}


import pytest


@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_train_model_various_sample_sizes(n_samples: int) -> None:
    import numpy as np
    import pandas as pd

    from app.model import train_model

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n_samples),
            "day_of_week": rng.integers(0, 7, n_samples),
            "month": rng.integers(1, 13, n_samples),
            "temperature_c": rng.uniform(5, 35, n_samples),
            "humidity_pct": rng.uniform(30, 80, n_samples),
            "occupancy": rng.integers(0, 150, n_samples),
            "hvac_state": rng.integers(0, 2, n_samples),
        }
    )
    y = pd.Series(rng.uniform(5, 30, n_samples))
    model, metrics = train_model(X, y)
    assert model is not None
    assert "mae" in metrics


@pytest.mark.parametrize("metric", ["mae", "rmse", "r2", "cv_r2_mean"])
def test_train_model_returns_expected_metrics(metric: str) -> None:
    import numpy as np
    import pandas as pd

    from app.model import train_model

    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(5, 35, n),
            "humidity_pct": rng.uniform(30, 80, n),
            "occupancy": rng.integers(0, 150, n),
            "hvac_state": rng.integers(0, 2, n),
        }
    )
    y = pd.Series(rng.uniform(5, 30, n))
    _, metrics = train_model(X, y)
    assert metric in metrics
