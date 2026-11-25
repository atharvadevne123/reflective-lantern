"""Extended feature engineering tests for temporal-pulse with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    LAG_STEPS,
    WINDOW_SIZES,
    add_lag_features,
    add_rate_of_change,
    add_rolling_statistics,
    add_time_features,
)


def _sensor_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "value": rng.normal(10, 2, n),
        "temp": rng.uniform(20, 30, n),
    })


@pytest.mark.parametrize("window", WINDOW_SIZES)
def test_rolling_stats_adds_expected_columns(window: int) -> None:
    df = _sensor_df(50)
    out = add_rolling_statistics(df, ["value"])
    for stat in ("roll_mean", "roll_std", "roll_min", "roll_max"):
        assert f"value_{stat}_{window}" in out.columns


@pytest.mark.parametrize("n", [1, 5, 50, 200])
def test_rolling_stats_preserves_row_count(n: int) -> None:
    df = _sensor_df(n)
    out = add_rolling_statistics(df, ["value"])
    assert out.shape[0] == n


def test_rolling_stats_no_nans_due_to_min_periods() -> None:
    df = _sensor_df(5)
    out = add_rolling_statistics(df, ["value"])
    for w in WINDOW_SIZES:
        assert not out[f"value_roll_mean_{w}"].isna().any()


def test_rolling_stats_mean_bounded_by_min_max() -> None:
    df = _sensor_df(50)
    out = add_rolling_statistics(df, ["value"])
    for w in WINDOW_SIZES:
        assert (out[f"value_roll_mean_{w}"] >= out[f"value_roll_min_{w}"]).all()
        assert (out[f"value_roll_mean_{w}"] <= out[f"value_roll_max_{w}"]).all()


def test_rolling_stats_constant_series_std_zero() -> None:
    df = pd.DataFrame({"value": [5.0] * 20})
    out = add_rolling_statistics(df, ["value"])
    for w in WINDOW_SIZES:
        assert (out[f"value_roll_std_{w}"] == pytest.approx(0.0, abs=1e-9)).all()


@pytest.mark.parametrize("lag", LAG_STEPS)
def test_lag_features_adds_expected_columns(lag: int) -> None:
    df = _sensor_df(50)
    out = add_lag_features(df, ["value"])
    assert f"value_lag_{lag}" in out.columns


@pytest.mark.parametrize("n", [5, 50])
def test_lag_features_preserves_row_count(n: int) -> None:
    df = _sensor_df(n)
    out = add_lag_features(df, ["value"])
    assert out.shape[0] == n


def test_lag_features_no_nans() -> None:
    df = _sensor_df(20)
    out = add_lag_features(df, ["value"])
    for lag in LAG_STEPS:
        assert not out[f"value_lag_{lag}"].isna().any()


def test_rate_of_change_adds_roc_and_roc2() -> None:
    df = _sensor_df(20)
    out = add_rate_of_change(df, ["value"])
    assert "value_roc" in out.columns
    assert "value_roc2" in out.columns


def test_rate_of_change_no_nans() -> None:
    df = _sensor_df(20)
    out = add_rate_of_change(df, ["value"])
    assert not out["value_roc"].isna().any()
    assert not out["value_roc2"].isna().any()


def test_rate_of_change_constant_series_zero() -> None:
    df = pd.DataFrame({"value": [3.0] * 10})
    out = add_rate_of_change(df, ["value"])
    assert (out["value_roc"] == pytest.approx(0.0, abs=1e-9)).all()


def test_rate_of_change_linear_first_diff_constant() -> None:
    df = pd.DataFrame({"value": [float(i * 2) for i in range(10)]})
    out = add_rate_of_change(df, ["value"])
    assert (out["value_roc"].iloc[1:] == pytest.approx(2.0, abs=1e-9)).all()


def test_add_time_features_adds_cyclical_columns() -> None:
    import pandas as pd
    timestamps = pd.date_range("2024-01-01", periods=24, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "value": np.ones(24)})
    out = add_time_features(df, timestamp_col="timestamp")
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert col in out.columns


def test_add_time_features_cyclical_range() -> None:
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "value": np.ones(48)})
    out = add_time_features(df, timestamp_col="timestamp")
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert (out[col] >= -1.0).all()
        assert (out[col] <= 1.0).all()


def test_add_time_features_missing_timestamp_col_no_error() -> None:
    df = _sensor_df(10)
    out = add_time_features(df, timestamp_col="timestamp")
    assert "hour_sin" not in out.columns


@pytest.mark.parametrize("n_cols", [1, 2, 3])
def test_rolling_stats_multiple_columns(n_cols: int) -> None:
    cols = [f"sensor_{i}" for i in range(n_cols)]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.normal(0, 1, 30) for col in cols})
    out = add_rolling_statistics(df, cols)
    for col in cols:
        assert f"{col}_roll_mean_{WINDOW_SIZES[0]}" in out.columns
