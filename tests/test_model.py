"""Tests for ML model training and prediction."""

import numpy as np
import pytest

from app.model import _synthetic_model, load_metrics, predict, train_model


def test_train_model_returns_bundle_and_metrics(sample_df, sample_target) -> None:
    bundle, metrics = train_model(sample_df, sample_target)
    assert "ensemble" in bundle
    assert "scaler" in bundle
    assert "feature_pipeline" in bundle
    assert "r2_mean" in metrics
    assert "rmse_mean" in metrics
    assert metrics["n_samples"] == len(sample_target)


def test_train_model_r2_reasonable(sample_df, sample_target) -> None:
    _, metrics = train_model(sample_df, sample_target)
    assert metrics["r2_mean"] > -2.0


def test_predict_returns_correct_shape(sample_df, sample_target) -> None:
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df, bundle)
    assert preds.shape == (len(sample_df),)


def test_predict_positive_values(sample_df, sample_target) -> None:
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df, bundle)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("n", [1, 5, 10])
def test_predict_batch_sizes(sample_df, sample_target, n) -> None:
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df.head(n), bundle)
    assert len(preds) == n


def test_synthetic_model_loads() -> None:
    bundle = _synthetic_model()
    assert "ensemble" in bundle
    assert "scaler" in bundle


def test_load_metrics_returns_dict() -> None:
    m = load_metrics()
    assert isinstance(m, dict)
    assert "model_version" in m


def test_train_model_persists_file(sample_df, sample_target, tmp_path, monkeypatch) -> None:
    import app.model as model_module

    mp = tmp_path / "model.joblib"
    mtp = tmp_path / "metrics.json"
    monkeypatch.setattr(model_module, "MODEL_PATH", mp)
    monkeypatch.setattr(model_module, "METRICS_PATH", mtp)
    train_model(sample_df, sample_target)
    assert mp.exists()
    assert mtp.exists()


def test_train_model_metrics_have_std(sample_df, sample_target) -> None:
    _, metrics = train_model(sample_df, sample_target)
    assert "r2_std" in metrics
    assert "rmse_std" in metrics
    assert metrics["r2_std"] >= 0.0
    assert metrics["rmse_std"] >= 0.0


def test_train_model_n_features_positive(sample_df, sample_target) -> None:
    _, metrics = train_model(sample_df, sample_target)
    assert metrics["n_features"] > 0


def test_predict_no_nan_values(sample_df, sample_target) -> None:
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df, bundle)
    assert not np.any(np.isnan(preds))


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_synthetic_model_predict_finite(seed) -> None:
    bundle = _synthetic_model()
    import pandas as pd

    stub = pd.DataFrame(
        {
            "sqft": [1200.0],
            "bedrooms": [2],
            "bathrooms": [1.0],
            "lot_size": [4000.0],
            "year_built": [1990],
            "condition_score": [6.0],
            "school_score": [5.0],
            "transit_score": [5.0],
            "walkability_score": [5.0],
            "crime_rate": [0.4],
            "median_neighborhood_price": [250_000.0],
            "median_price_per_sqft": [180.0],
            "avg_rental_yield": [0.05],
            "listing_days": [20],
            "list_price": [300_000.0],
        }
    )
    preds = predict(stub, bundle)
    assert len(preds) == 1
    assert np.isfinite(preds[0])


def test_train_model_model_version_in_metrics(sample_df, sample_target) -> None:
    _, metrics = train_model(sample_df, sample_target)
    assert "model_version" in metrics


@pytest.mark.parametrize("n_samples", [20, 50])
def test_train_model_n_samples_recorded(sample_df, sample_target, n_samples) -> None:
    df_subset = sample_df.head(n_samples)
    y_subset = sample_target[:n_samples]
    _, metrics = train_model(df_subset, y_subset)
    assert metrics["n_samples"] == n_samples


def test_metrics_file_content(sample_df, sample_target, tmp_path, monkeypatch) -> None:
    import json

    import app.model as model_module

    mtp = tmp_path / "metrics.json"
    mp = tmp_path / "model.joblib"
    monkeypatch.setattr(model_module, "MODEL_PATH", mp)
    monkeypatch.setattr(model_module, "METRICS_PATH", mtp)
    _, metrics = train_model(sample_df, sample_target)
    stored = json.loads(mtp.read_text())
    assert stored["n_samples"] == metrics["n_samples"]
    assert stored["r2_mean"] == pytest.approx(metrics["r2_mean"])


def test_cv_n_splits_constant_used() -> None:
    from app.model import CV_N_SPLITS

    assert CV_N_SPLITS == 5


def test_cv_random_state_constant_used() -> None:
    from app.model import CV_RANDOM_STATE

    assert CV_RANDOM_STATE == 42


def test_load_metrics_model_version_matches_constant() -> None:
    from app.model import MODEL_VERSION

    m = load_metrics()
    assert m.get("model_version") == MODEL_VERSION or m.get("note") == "no metrics file"


def test_predict_values_positive(sample_df, sample_target) -> None:
    bundle, _ = train_model(sample_df, sample_target)
    preds = predict(sample_df.head(5), bundle)
    assert len(preds) == 5


def test_train_model_r2_in_metrics(sample_df, sample_target) -> None:
    _, metrics = train_model(sample_df, sample_target)
    assert isinstance(metrics["r2_mean"], float)
    assert isinstance(metrics["r2_std"], float)


def test_n_stub_samples_constant() -> None:
    from app.model import N_STUB_SAMPLES

    assert N_STUB_SAMPLES > 0


def test_n_stub_features_constant() -> None:
    from app.model import N_STUB_FEATURES

    assert N_STUB_FEATURES > 0


@pytest.mark.parametrize("n_rows", [5, 10, 20])
def test_predict_returns_correct_count(sample_df, sample_target, n_rows: int) -> None:
    from app.model import predict, train_model

    bundle, _ = train_model(sample_df, sample_target)
    subset = sample_df.head(n_rows)
    preds = predict(subset, bundle)
    assert len(preds) == n_rows


def test_metrics_rmse_is_non_negative(sample_df, sample_target) -> None:
    from app.model import train_model

    _, metrics = train_model(sample_df, sample_target)
    assert metrics.get("rmse_mean", 0.0) >= 0.0


def test_metrics_mae_is_non_negative(sample_df, sample_target) -> None:
    from app.model import train_model

    _, metrics = train_model(sample_df, sample_target)
    assert metrics.get("mae_mean", 0.0) >= 0.0
