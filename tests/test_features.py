"""Tests for feature engineering pipeline."""

from __future__ import annotations

import pytest

from app.features import (
    DropColumnsTransformer,
    LagFeatureTransformer,
    PeakPeriodEncoder,
    RatioFeatureTransformer,
    RollingStatsTransformer,
    TemporalFeatureTransformer,
    build_feature_pipeline,
    build_raw_feature_df,
)


class TestLagFeatureTransformer:
    def test_creates_lag_columns(self):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, [3000.0] * 200)
        t = LagFeatureTransformer()
        out = t.fit_transform(df)
        assert "lag_1h" in out.columns
        assert "lag_24h" in out.columns
        assert "lag_168h" in out.columns

    def test_lag_values_from_history(self):
        history = list(range(200, 370))
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, history)
        out = LagFeatureTransformer().fit_transform(df)
        assert out["lag_1h"].iloc[0] == pytest.approx(history[-1], rel=0.01)
        assert out["lag_24h"].iloc[0] == pytest.approx(history[-24], rel=0.01)

    def test_no_history_fills_zeros(self):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, None)
        out = LagFeatureTransformer().fit_transform(df)
        assert out["lag_1h"].iloc[0] == 0.0
        assert out["lag_24h"].iloc[0] == 0.0

    @pytest.mark.parametrize("lags", [[1], [1, 24], [1, 24, 168]])
    def test_custom_lags(self, lags):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, [3000.0] * 200)
        out = LagFeatureTransformer(lags=lags).fit_transform(df)
        for lag in lags:
            assert f"lag_{lag}h" in out.columns


class TestRollingStatsTransformer:
    def test_creates_rolling_columns(self):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, [4000.0] * 30)
        out = RollingStatsTransformer().fit_transform(df)
        assert "rolling_mean_3h" in out.columns
        assert "rolling_std_3h" in out.columns
        assert "rolling_mean_24h" in out.columns

    def test_rolling_mean_correct(self):
        history = [float(i) for i in range(1, 31)]  # 1..30
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, history)
        out = RollingStatsTransformer(windows=[3]).fit_transform(df)
        expected_mean_3 = (28.0 + 29.0 + 30.0) / 3
        assert out["rolling_mean_3h"].iloc[0] == pytest.approx(expected_mean_3, rel=0.01)


class TestTemporalFeatureTransformer:
    def test_creates_sin_cos_columns(self):
        df = build_raw_feature_df(12, 3, 6, False, 20.0, 50.0)
        out = TemporalFeatureTransformer().fit_transform(df)
        assert "hour_sin" in out.columns
        assert "hour_cos" in out.columns
        assert "dow_sin" in out.columns
        assert "month_sin" in out.columns

    @pytest.mark.parametrize("hour,expected_sin", [(0, 0.0), (6, 1.0), (12, 0.0), (18, -1.0)])
    def test_hour_sin_values(self, hour, expected_sin):
        df = build_raw_feature_df(hour, 0, 1, False, 20.0, 50.0)
        out = TemporalFeatureTransformer().fit_transform(df)
        assert out["hour_sin"].iloc[0] == pytest.approx(expected_sin, abs=0.01)


class TestPeakPeriodEncoder:
    def test_peak_hour_detection(self):
        df = build_raw_feature_df(14, 2, 6, False, 25.0, 60.0)
        out = PeakPeriodEncoder().fit_transform(df)
        assert out["is_peak_hour"].iloc[0] == 1

    def test_off_peak_hour(self):
        df = build_raw_feature_df(3, 2, 6, False, 25.0, 60.0)
        out = PeakPeriodEncoder().fit_transform(df)
        assert out["is_peak_hour"].iloc[0] == 0

    def test_morning_peak_detection(self):
        df = build_raw_feature_df(8, 0, 1, False, 15.0, 55.0)
        out = PeakPeriodEncoder().fit_transform(df)
        assert out["is_morning_peak"].iloc[0] == 1

    def test_evening_peak_detection(self):
        df = build_raw_feature_df(19, 0, 1, False, 25.0, 65.0)
        out = PeakPeriodEncoder().fit_transform(df)
        assert out["is_evening_peak"].iloc[0] == 1


class TestRatioFeatureTransformer:
    def test_creates_heat_index(self):
        df = build_raw_feature_df(12, 2, 7, False, 30.0, 80.0)
        out = RatioFeatureTransformer().fit_transform(df)
        assert "heat_index" in out.columns
        assert "cooling_demand_proxy" in out.columns
        assert "heating_demand_proxy" in out.columns

    def test_cooling_demand_positive_when_hot(self):
        df = build_raw_feature_df(12, 2, 7, False, 35.0, 70.0)
        out = RatioFeatureTransformer().fit_transform(df)
        assert out["cooling_demand_proxy"].iloc[0] > 0

    def test_heating_demand_positive_when_cold(self):
        df = build_raw_feature_df(12, 2, 1, False, -5.0, 50.0)
        out = RatioFeatureTransformer().fit_transform(df)
        assert out["heating_demand_proxy"].iloc[0] > 0


class TestDropColumnsTransformer:
    def test_drops_helper_columns(self):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0, [3000.0] * 5, "northeast")
        out = DropColumnsTransformer().fit_transform(df)
        assert "historical_loads" not in out.columns
        assert "region" not in out.columns


class TestBuildFeaturePipeline:
    def test_pipeline_runs_end_to_end(self):
        from app.model import generate_synthetic_training_data
        X, y = generate_synthetic_training_data(n_samples=50)
        pipe = build_feature_pipeline()
        # Pipeline expects full numeric df; test with synthetic data directly
        result = pipe.fit_transform(X)
        assert result.shape[0] == 50
        assert result.shape[1] > 0

    def test_raw_feature_df_shape(self):
        df = build_raw_feature_df(12, 2, 6, False, 25.0, 60.0)
        assert df.shape[0] == 1
        assert "hour" in df.columns
        assert "temperature_c" in df.columns
