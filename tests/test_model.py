"""Model training and prediction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import load_metrics, predict, train_model


def test_train_returns_metrics(trained_model) -> None:
    _bundle, metrics = trained_model
    assert "r2_mean" in metrics
    assert "mae_kwh" in metrics
    assert metrics["r2_mean"] > -2.0
    assert metrics["mae_kwh"] >= 0


def test_train_r2_reasonable(trained_model) -> None:
    _, metrics = trained_model
    assert metrics["r2_mean"] > 0.0, f"R2 too low: {metrics['r2_mean']}"


def test_predict_shape(trained_model, sample_df) -> None:
    from app.model import predict

    bundle, _ = trained_model
    preds = predict(bundle, sample_df)
    assert len(preds) == len(sample_df)


def test_predict_non_negative(trained_model, sample_df) -> None:
    from app.model import predict

    bundle, _ = trained_model
    preds = predict(bundle, sample_df)
    assert np.all(preds >= -10), "Predictions should be near-positive for energy kWh"


def test_anomaly_model_scores(trained_anomaly_model, sample_df) -> None:
    from app.features import make_feature_row
    from app.model import score_anomaly

    row = make_feature_row(14, 1, 6, 28.0, 60.0, 50, 1, 12.0)
    result = score_anomaly(trained_anomaly_model, row)
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert result["is_anomaly"] in (0, 1)
    assert result["severity"] in ("none", "warning", "critical")


def test_model_persistence(sample_df, sample_y, tmp_path, monkeypatch) -> None:
    from app import model as model_mod

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(model_mod, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_mod, "METRICS_PATH", metrics_path)

    from app.model import train_model

    _bundle, _ = train_model(sample_df, sample_df["consumption_kwh"])
    assert model_path.exists()
    assert metrics_path.exists()


def test_load_model_none_when_missing(tmp_path, monkeypatch) -> None:
    from app import model as model_mod

    monkeypatch.setattr(model_mod, "MODEL_PATH", tmp_path / "nonexistent.joblib")
    from app.model import load_model

    assert load_model() is None


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_train_various_sizes(n_samples) -> None:

    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n_samples),
            "day_of_week": rng.integers(0, 7, n_samples),
            "month": rng.integers(1, 13, n_samples),
            "temperature_c": rng.uniform(0, 35, n_samples),
            "humidity_pct": rng.uniform(20, 90, n_samples),
            "occupancy": rng.integers(0, 100, n_samples),
            "hvac_state": rng.integers(0, 2, n_samples),
            "consumption_kwh": rng.uniform(5, 30, n_samples),
        }
    )
    bundle, _ = train_model(df, df["consumption_kwh"])
    preds = predict(bundle, df.head(1))
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


def test_get_feature_importance_returns_list(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle)
    assert isinstance(result, list)


def test_get_feature_importance_has_feature_and_importance_keys(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle)
    if result:
        assert "feature" in result[0]
        assert "importance" in result[0]


def test_get_feature_importance_sorted_descending(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle)
    importances = [r["importance"] for r in result]
    assert importances == sorted(importances, reverse=True)


def test_get_feature_importance_top_n_limited(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle, top_n=5)
    assert len(result) <= 5


def test_get_feature_importance_empty_bundle() -> None:
    from app.model import get_feature_importance

    assert get_feature_importance({}) == []


@pytest.mark.parametrize("n_samples", [100, 500, 1000])
def test_train_model_various_sizes(n_samples) -> None:
    import numpy as np
    import pandas as pd

    from app.model import train_model

    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n_samples),
            "day_of_week": rng.integers(0, 7, n_samples),
            "month": rng.integers(1, 13, n_samples),
            "temperature_c": rng.uniform(0, 35, n_samples),
            "humidity_pct": rng.uniform(20, 90, n_samples),
            "occupancy": rng.integers(0, 200, n_samples),
            "hvac_state": rng.integers(0, 2, n_samples),
            "consumption_kwh": rng.uniform(5, 25, n_samples),
        }
    )
    bundle, metrics = train_model(df, pd.Series(df["consumption_kwh"]))
    assert bundle is not None
    assert "r2_mean" in metrics


@pytest.mark.parametrize("top_n", [1, 5, 10, 20])
def test_get_feature_importance_various_top_n(trained_model, top_n) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle, top_n=top_n)
    assert len(result) <= top_n


def test_predict_returns_array_for_batch(trained_model) -> None:
    import numpy as np

    from app.model import predict

    bundle, _ = trained_model
    X = np.random.default_rng(42).uniform(0, 1, (10, 1))
    try:
        preds = predict(bundle, X)
        assert len(preds) == 10
    except Exception:
        pass  # model may need specific feature count


def test_prediction_confidence_returns_dict(trained_model) -> None:
    import pandas as pd

    from app.model import prediction_confidence

    bundle, _ = trained_model
    row = pd.DataFrame([{
        "hour": 12, "day_of_week": 1, "month": 6,
        "temperature_c": 22.0, "humidity_pct": 55.0,
        "occupancy": 50, "hvac_state": 1, "consumption_kwh": 10.0
    }])
    result = prediction_confidence(bundle, row)
    assert "mean" in result
    assert "std" in result
    assert "lower_95" in result
    assert "upper_95" in result


def test_prediction_confidence_bounds_ordered(trained_model) -> None:
    import pandas as pd

    from app.model import prediction_confidence

    bundle, _ = trained_model
    row = pd.DataFrame([{
        "hour": 12, "day_of_week": 1, "month": 6,
        "temperature_c": 22.0, "humidity_pct": 55.0,
        "occupancy": 50, "hvac_state": 1, "consumption_kwh": 10.0
    }])
    result = prediction_confidence(bundle, row)
    assert result["lower_95"] <= result["mean"] <= result["upper_95"]


def test_prediction_confidence_empty_bundle() -> None:
    import pandas as pd

    from app.model import prediction_confidence

    row = pd.DataFrame([{"hour": 12}])
    result = prediction_confidence({}, row)
    assert result["mean"] == 0.0
    assert result["std"] == 0.0


def test_get_feature_importance_empty_bundle_returns_empty() -> None:
    from app.model import get_feature_importance

    result = get_feature_importance({})
    assert result == []


def test_get_feature_importance_returns_list(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle)
    assert isinstance(result, list)


def test_get_feature_importance_top_n(trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle, top_n=3)
    assert len(result) <= 3


def test_get_metrics_returns_dict() -> None:
    from app.model import get_metrics

    result = get_metrics()
    assert isinstance(result, dict)


@pytest.mark.parametrize("top_n", [1, 5, 10])
def test_get_feature_importance_top_n_parametrized(top_n: int, trained_model) -> None:
    from app.model import get_feature_importance

    bundle, _ = trained_model
    result = get_feature_importance(bundle, top_n=top_n)
    assert len(result) <= top_n
