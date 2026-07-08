"""Tests for feature engineering pipeline in Temporal-Pulse."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_series(n: int = 30) -> pd.DataFrame:
    """Return a simple two-channel sensor DataFrame."""
    rng = np.random.default_rng(0)
    base = datetime(2026, 1, 1)
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(minutes=i) for i in range(n)],
            "sensor_id": ["s1"] * n,
            "temp": rng.normal(20, 1, n).tolist(),
            "pressure": rng.normal(1000, 5, n).tolist(),
        }
    )


class TestRollingStatistics:
    def test_adds_rolling_mean_columns(self):
        from app.features import add_rolling_statistics
        df = make_series()
        result = add_rolling_statistics(df, ["temp"])
        assert "temp_roll_mean_5" in result.columns
        assert "temp_roll_mean_10" in result.columns

    def test_adds_rolling_std_columns(self):
        from app.features import add_rolling_statistics
        df = make_series()
        result = add_rolling_statistics(df, ["temp"])
        assert "temp_roll_std_5" in result.columns

    def test_no_nan_in_rolling_mean(self):
        from app.features import add_rolling_statistics
        df = make_series()
        result = add_rolling_statistics(df, ["temp"])
        assert not result["temp_roll_mean_5"].isna().any()

    def test_original_columns_preserved(self):
        from app.features import add_rolling_statistics
        df = make_series()
        result = add_rolling_statistics(df, ["temp"])
        assert "temp" in result.columns


class TestLagFeatures:
    def test_adds_lag_columns(self):
        from app.features import add_lag_features
        df = make_series()
        result = add_lag_features(df, ["temp"])
        assert "temp_lag_1" in result.columns
        assert "temp_lag_3" in result.columns

    def test_no_nan_after_lag_fill(self):
        from app.features import add_lag_features
        df = make_series()
        result = add_lag_features(df, ["temp"])
        assert not result["temp_lag_1"].isna().any()

    @pytest.mark.parametrize("lag", [1, 2, 3, 5])
    def test_each_lag_present(self, lag):
        from app.features import add_lag_features
        df = make_series()
        result = add_lag_features(df, ["pressure"])
        assert f"pressure_lag_{lag}" in result.columns


class TestRateOfChange:
    def test_adds_roc_column(self):
        from app.features import add_rate_of_change
        df = make_series()
        result = add_rate_of_change(df, ["temp"])
        assert "temp_roc" in result.columns

    def test_adds_second_order_roc(self):
        from app.features import add_rate_of_change
        df = make_series()
        result = add_rate_of_change(df, ["temp"])
        assert "temp_roc2" in result.columns


class TestTimeFeatures:
    def test_adds_hour_sin_cos(self):
        from app.features import add_time_features
        df = make_series()
        result = add_time_features(df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_adds_day_of_week_features(self):
        from app.features import add_time_features
        df = make_series()
        result = add_time_features(df)
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns

    def test_time_features_bounded(self):
        from app.features import add_time_features
        df = make_series()
        result = add_time_features(df)
        assert result["hour_sin"].between(-1.0, 1.0).all()
        assert result["hour_cos"].between(-1.0, 1.0).all()


class TestBuildFeatureMatrix:
    def test_returns_dataframe_and_cols(self, sample_readings):
        from app.features import build_feature_matrix
        raw = [
            {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
            for r in sample_readings
        ]
        df, feature_cols = build_feature_matrix(raw)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(feature_cols, list)

    def test_feature_cols_non_empty(self, sample_readings):
        from app.features import build_feature_matrix
        raw = [
            {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
            for r in sample_readings
        ]
        _, feature_cols = build_feature_matrix(raw)
        assert len(feature_cols) > 0

    def test_no_nan_in_feature_matrix(self, sample_readings):
        from app.features import build_feature_matrix
        raw = [
            {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
            for r in sample_readings
        ]
        df, feature_cols = build_feature_matrix(raw)
        assert not df[feature_cols].isna().any().any()

    def test_sklearn_pipeline_fits(self, feature_matrix):
        from app.features import build_sklearn_pipeline
        import numpy as np
        df, feature_cols = feature_matrix
        X = df[feature_cols].to_numpy(dtype=np.float32)
        pipeline = build_sklearn_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape
