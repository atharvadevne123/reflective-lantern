"""Tests for feature engineering pipeline transformers."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "consumption_kwh": [3.0, 4.5, 2.8, 6.1, 5.0],
        "temperature_c": [15.0, 22.0, 8.0, 35.0, 20.0],
        "humidity_pct": [50.0, 65.0, 45.0, 80.0, 55.0],
        "hour_of_day": [8, 14, 2, 18, 12],
        "day_of_week": [0, 2, 5, 3, 1],
        "is_holiday": [0, 0, 0, 0, 1],
        "building_type": ["residential", "commercial", "residential", "industrial", "office"],
    })


class TestLagFeatureTransformer:
    def test_creates_lag_columns(self, sample_df):
        from app.features import LagFeatureTransformer

        t = LagFeatureTransformer(lags=[1, 2])
        out = t.fit_transform(sample_df)
        assert "consumption_lag_1h" in out.columns
        assert "consumption_lag_2h" in out.columns

    def test_lag_values_are_non_negative(self, sample_df):
        from app.features import LagFeatureTransformer

        out = LagFeatureTransformer(lags=[1]).fit_transform(sample_df)
        assert (out["consumption_lag_1h"] >= 0).all()

    def test_fit_returns_self(self, sample_df):
        from app.features import LagFeatureTransformer

        t = LagFeatureTransformer()
        assert t.fit(sample_df) is t


class TestRollingStatsTransformer:
    def test_creates_rolling_columns(self, sample_df):
        from app.features import RollingStatsTransformer

        out = RollingStatsTransformer(windows=[3]).fit_transform(sample_df)
        assert "rolling_mean_3h" in out.columns
        assert "rolling_std_3h" in out.columns
        assert "rolling_max_3h" in out.columns

    def test_rolling_mean_equals_manual(self, sample_df):
        from app.features import RollingStatsTransformer

        out = RollingStatsTransformer(windows=[3]).fit_transform(sample_df)
        expected = sample_df["consumption_kwh"].rolling(3, min_periods=1).mean()
        pd.testing.assert_series_equal(out["rolling_mean_3h"], expected, check_names=False)


class TestTemporalFeatureTransformer:
    def test_creates_cyclical_features(self, sample_df):
        from app.features import TemporalFeatureTransformer

        out = TemporalFeatureTransformer().fit_transform(sample_df)
        assert "hour_sin" in out.columns
        assert "hour_cos" in out.columns
        assert "day_sin" in out.columns
        assert "day_cos" in out.columns

    def test_cyclical_values_in_range(self, sample_df):
        from app.features import TemporalFeatureTransformer

        out = TemporalFeatureTransformer().fit_transform(sample_df)
        assert out["hour_sin"].between(-1, 1).all()
        assert out["hour_cos"].between(-1, 1).all()

    def test_peak_flags_binary(self, sample_df):
        from app.features import TemporalFeatureTransformer

        out = TemporalFeatureTransformer().fit_transform(sample_df)
        assert set(out["is_peak_morning"].unique()).issubset({0, 1})
        assert set(out["is_weekend"].unique()).issubset({0, 1})

    def test_weekend_flag_correct(self, sample_df):
        from app.features import TemporalFeatureTransformer

        out = TemporalFeatureTransformer().fit_transform(sample_df)
        weekend_rows = sample_df["day_of_week"] >= 5
        assert (out.loc[weekend_rows, "is_weekend"] == 1).all()


class TestWeatherRatioTransformer:
    def test_creates_degree_features(self, sample_df):
        from app.features import WeatherRatioTransformer

        out = WeatherRatioTransformer().fit_transform(sample_df)
        assert "cooling_degree" in out.columns
        assert "heating_degree" in out.columns
        assert "temp_deviation" in out.columns

    def test_cooling_non_negative(self, sample_df):
        from app.features import WeatherRatioTransformer

        out = WeatherRatioTransformer().fit_transform(sample_df)
        assert (out["cooling_degree"] >= 0).all()

    def test_heating_non_negative(self, sample_df):
        from app.features import WeatherRatioTransformer

        out = WeatherRatioTransformer().fit_transform(sample_df)
        assert (out["heating_degree"] >= 0).all()


class TestBuildingEncoderTransformer:
    def test_replaces_building_type_with_intensity(self, sample_df):
        from app.features import BuildingEncoderTransformer

        out = BuildingEncoderTransformer().fit_transform(sample_df)
        assert "building_intensity" in out.columns
        assert "building_type" not in out.columns

    def test_industrial_higher_than_residential(self, sample_df):
        from app.features import BuildingEncoderTransformer

        out = BuildingEncoderTransformer().fit_transform(sample_df)
        res_rows = sample_df["building_type"] == "residential"
        ind_rows = sample_df["building_type"] == "industrial"
        assert out.loc[res_rows, "building_intensity"].iloc[0] < out.loc[ind_rows, "building_intensity"].iloc[0]


class TestBuildFeaturePipeline:
    def test_pipeline_runs_end_to_end(self, sample_df):
        from app.features import build_feature_pipeline

        pipe = build_feature_pipeline()
        X = pipe.fit_transform(sample_df)
        assert X.shape[0] == len(sample_df)

    def test_pipeline_output_no_raw_columns(self, sample_df):
        from app.features import build_feature_pipeline

        pipe = build_feature_pipeline()
        pipe.fit(sample_df)
        result = pipe.transform(sample_df)
        if hasattr(result, "columns"):
            assert "consumption_kwh" not in result.columns
            assert "building_type" not in result.columns
