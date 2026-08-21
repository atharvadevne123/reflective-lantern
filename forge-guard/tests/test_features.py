"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    LagFeatureTransformer,
    PolynomialSensorTransformer,
    RatioFeatureTransformer,
    RollingStatsTransformer,
    build_feature_pipeline,
    engineer_single,
    generate_synthetic_data,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [70.0, 75.0, 80.0, 85.0, 90.0],
            "pressure": [50.0, 52.0, 48.0, 55.0, 60.0],
            "vibration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cycle_time": [30.0, 29.0, 31.0, 28.0, 32.0],
            "tool_wear": [10.0, 12.0, 14.0, 16.0, 18.0],
            "power_consumption": [90.0, 95.0, 100.0, 105.0, 110.0],
            "humidity": [40.0, 42.0, 44.0, 46.0, 48.0],
        }
    )


def test_lag_transformer_adds_columns(small_df):
    t = LagFeatureTransformer(lags=2)
    out = t.fit_transform(small_df)
    assert "temperature_lag1" in out.columns
    assert "temperature_lag2" in out.columns
    assert len(out) == len(small_df)


def test_rolling_transformer_adds_columns(small_df):
    t = RollingStatsTransformer(window=3)
    out = t.fit_transform(small_df)
    assert "temperature_roll_mean" in out.columns
    assert "temperature_roll_std" in out.columns


def test_ratio_transformer_adds_derived_features(small_df):
    t = RatioFeatureTransformer()
    out = t.fit_transform(small_df)
    assert "temp_pressure_ratio" in out.columns
    assert "vibration_per_cycle" in out.columns
    assert "wear_rate" in out.columns
    assert "power_efficiency" in out.columns


def test_polynomial_transformer_adds_squared(small_df):
    t = PolynomialSensorTransformer()
    out = t.fit_transform(small_df)
    assert "vibration_sq" in out.columns
    assert "tool_wear_sq" in out.columns
    assert "temperature_sq" in out.columns


def test_build_feature_pipeline_produces_array(small_df):
    pipe = build_feature_pipeline()
    X = pipe.fit_transform(small_df)
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(small_df)
    assert X.shape[1] > len(small_df.columns)


def test_engineer_single_returns_2d_array():
    row = {
        "temperature": 78.5,
        "pressure": 52.0,
        "vibration": 2.1,
        "cycle_time": 28.0,
        "tool_wear": 15.0,
        "power_consumption": 98.0,
        "humidity": 45.0,
    }
    result = engineer_single(row)
    assert result.ndim == 2
    assert result.shape[0] == 1


def test_generate_synthetic_data_shape():
    df = generate_synthetic_data(n_samples=100, seed=1)
    assert len(df) == 100
    assert "defect" in df.columns
    assert df["defect"].isin([0, 1]).all()


def test_generate_synthetic_data_defect_rate():
    df = generate_synthetic_data(n_samples=5000, seed=42)
    rate = df["defect"].mean()
    assert 0.05 <= rate <= 0.60


@pytest.mark.parametrize("lags", [1, 2, 3])
def test_lag_count_parametrized(small_df, lags):
    t = LagFeatureTransformer(lags=lags)
    out = t.fit_transform(small_df)
    for lag in range(1, lags + 1):
        assert f"temperature_lag{lag}" in out.columns


def test_engineer_single_missing_column_defaults_to_zero():
    row = {"temperature": 78.5, "pressure": 52.0, "vibration": 2.1, "cycle_time": 28.0}
    result = engineer_single(row)
    assert result.ndim == 2
    assert not np.isnan(result).any()


def test_ratio_transformer_handles_zero_denominator():
    df = pd.DataFrame(
        {
            "temperature": [75.0],
            "pressure": [0.0],
            "vibration": [2.0],
            "cycle_time": [0.0],
            "tool_wear": [10.0],
            "power_consumption": [100.0],
            "humidity": [45.0],
        }
    )
    t = RatioFeatureTransformer()
    out = t.fit_transform(df)
    assert np.isfinite(out["temp_pressure_ratio"]).all()
    assert np.isfinite(out["vibration_per_cycle"]).all()


def test_pipeline_output_has_no_nan(small_df):
    pipe = build_feature_pipeline()
    X = pipe.fit_transform(small_df)
    assert not np.isnan(X).any()


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_synthetic_data_reproducible(seed):
    a = generate_synthetic_data(n_samples=50, seed=seed)
    b = generate_synthetic_data(n_samples=50, seed=seed)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("window", [3, 5, 10])
def test_rolling_window_parametrized(small_df, window):
    t = RollingStatsTransformer(window=window)
    out = t.fit_transform(small_df)
    assert "temperature_roll_mean" in out.columns
    assert not out["temperature_roll_mean"].isna().any()


def test_engineer_single_all_sensors_present():
    row = {
        "temperature": 75.0,
        "pressure": 50.0,
        "vibration": 2.1,
        "cycle_time": 28.0,
        "tool_wear": 15.0,
        "power_consumption": 98.0,
        "humidity": 45.0,
    }
    result = engineer_single(row)
    assert result.shape[0] == 1
    assert result.shape[1] >= 7


@pytest.mark.parametrize("n_samples", [50, 200, 500])
def test_synthetic_data_various_sizes(n_samples):
    df = generate_synthetic_data(n_samples=n_samples, seed=7)
    assert len(df) == n_samples
    assert df["defect"].isin([0, 1]).all()


def test_polynomial_transformer_values_correct(small_df):
    t = PolynomialSensorTransformer()
    out = t.fit_transform(small_df)
    expected = small_df["vibration"] ** 2
    pd.testing.assert_series_equal(
        out["vibration_sq"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_pipeline_fit_and_transform_shape_consistent(small_df):
    pipe = build_feature_pipeline()
    X_fit = pipe.fit_transform(small_df)
    # transform again should give same shape
    X_t = pipe.transform(small_df)
    assert X_fit.shape == X_t.shape


def test_engineer_single_no_nan_on_extreme_values():
    row = {
        "temperature": 299.9,
        "pressure": 199.9,
        "vibration": 49.9,
        "cycle_time": 599.9,
        "tool_wear": 499.9,
        "power_consumption": 499.9,
        "humidity": 99.9,
    }
    result = engineer_single(row)
    assert not np.isnan(result).any()


def test_lag_transformer_is_idempotent_on_fit(small_df):
    t = LagFeatureTransformer(lags=1)
    t.fit(small_df)
    out1 = t.transform(small_df)
    t.fit(small_df)
    out2 = t.transform(small_df)
    pd.testing.assert_frame_equal(out1, out2)


def test_rolling_stats_no_nan_in_std(small_df):
    t = RollingStatsTransformer(window=3)
    out = t.fit_transform(small_df)
    assert not out["temperature_roll_std"].isna().any()


def test_generate_synthetic_data_has_both_classes():
    df = generate_synthetic_data(n_samples=1000, seed=0)
    assert df["defect"].nunique() == 2


def test_pipeline_transform_without_refit_raises_or_works(small_df):
    pipe = build_feature_pipeline()
    pipe.fit(small_df)
    out = pipe.transform(small_df)
    assert isinstance(out, np.ndarray)


class TestSensorRangeFeature:
    def test_empty_row_returns_zeros(self) -> None:
        from app.features import sensor_range_feature

        result = sensor_range_feature({})
        assert result == {"sensor_min": 0.0, "sensor_max": 0.0, "sensor_range": 0.0}

    def test_single_sensor(self) -> None:
        from app.features import sensor_range_feature

        result = sensor_range_feature({"temperature": 75.0})
        assert result["sensor_min"] == pytest.approx(75.0)
        assert result["sensor_max"] == pytest.approx(75.0)
        assert result["sensor_range"] == pytest.approx(0.0)

    def test_range_computed_correctly(self) -> None:
        from app.features import sensor_range_feature

        result = sensor_range_feature({"a": 10.0, "b": 30.0, "c": 20.0})
        assert result["sensor_min"] == pytest.approx(10.0)
        assert result["sensor_max"] == pytest.approx(30.0)
        assert result["sensor_range"] == pytest.approx(20.0)


class TestDefectRiskIndex:
    def test_normal_readings_low_risk(self) -> None:
        from app.features import defect_risk_index

        risk = defect_risk_index(70.0, 2.0, 10.0, 45.0)
        assert risk == pytest.approx(0.0)

    def test_all_thresholds_exceeded(self) -> None:
        from app.features import defect_risk_index

        risk = defect_risk_index(90.0, 6.0, 50.0, 60.0)
        assert risk == pytest.approx(0.95)

    def test_result_in_range(self) -> None:
        from app.features import defect_risk_index

        risk = defect_risk_index(100.0, 10.0, 100.0, 100.0)
        assert 0.0 <= risk <= 1.0

    @pytest.mark.parametrize(
        "temp,vib,wear,pres,expected",
        [
            (70.0, 2.0, 10.0, 45.0, 0.0),
            (90.0, 2.0, 10.0, 45.0, 0.30),
            (70.0, 6.0, 10.0, 45.0, 0.25),
        ],
    )
    def test_parametrized(self, temp, vib, wear, pres, expected) -> None:
        from app.features import defect_risk_index

        assert defect_risk_index(temp, vib, wear, pres) == pytest.approx(expected)

    @pytest.mark.parametrize("temp", [60.0, 70.0, 80.0, 90.0, 100.0])
    def test_risk_increases_with_temperature(self, temp: float) -> None:
        from app.features import defect_risk_index

        risk = defect_risk_index(temp, 2.0, 10.0, 45.0)
        assert 0.0 <= risk <= 1.0

    def test_all_normal_values_give_zero_risk(self) -> None:
        from app.features import defect_risk_index

        assert defect_risk_index(70.0, 2.0, 10.0, 45.0) == pytest.approx(0.0)

    def test_all_max_values_give_high_risk(self) -> None:
        from app.features import defect_risk_index

        risk = defect_risk_index(150.0, 20.0, 100.0, 120.0)
        assert risk >= 0.5
