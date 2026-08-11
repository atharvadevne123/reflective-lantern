"""Tests for feature engineering pipeline in Temporal-Pulse."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

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
        import numpy as np

        from app.features import build_sklearn_pipeline

        df, feature_cols = feature_matrix
        X = df[feature_cols].to_numpy(dtype=np.float32)
        pipeline = build_sklearn_pipeline()
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == X.shape


class TestSensorZscoreRow:
    def test_empty_row_returns_empty(self) -> None:
        from app.features import sensor_zscore_row

        assert sensor_zscore_row({}) == {}

    def test_constant_values_returns_zeros(self) -> None:
        from app.features import sensor_zscore_row

        result = sensor_zscore_row({"a": 5.0, "b": 5.0, "c": 5.0})
        assert all(v == pytest.approx(0.0) for v in result.values())

    def test_output_has_zscore_suffix(self) -> None:
        from app.features import sensor_zscore_row

        result = sensor_zscore_row({"x": 1.0, "y": 3.0})
        assert "x_zscore" in result
        assert "y_zscore" in result

    def test_opposite_values_opposite_zscores(self) -> None:
        from app.features import sensor_zscore_row

        result = sensor_zscore_row({"a": 0.0, "b": 10.0})
        assert result["a_zscore"] == pytest.approx(-result["b_zscore"])

    @pytest.mark.parametrize("n_sensors", [1, 2, 5])
    def test_output_length_matches_input(self, n_sensors) -> None:
        from app.features import sensor_zscore_row

        row = {f"s{i}": float(i) for i in range(n_sensors)}
        result = sensor_zscore_row(row)
        assert len(result) == n_sensors


class TestThresholdExceedanceVector:
    def test_no_thresholds_returns_empty(self) -> None:
        from app.features import threshold_exceedance_vector

        result = threshold_exceedance_vector({"a": 1.0}, {})
        assert result == {}

    def test_exceeded_returns_one(self) -> None:
        from app.features import threshold_exceedance_vector

        result = threshold_exceedance_vector({"temp": 100.0}, {"temp": 90.0})
        assert result["temp_exceeded"] == 1

    def test_not_exceeded_returns_zero(self) -> None:
        from app.features import threshold_exceedance_vector

        result = threshold_exceedance_vector({"temp": 80.0}, {"temp": 90.0})
        assert result["temp_exceeded"] == 0

    def test_sensor_absent_from_thresholds_skipped(self) -> None:
        from app.features import threshold_exceedance_vector

        result = threshold_exceedance_vector({"a": 5.0, "b": 5.0}, {"a": 3.0})
        assert "b_exceeded" not in result
        assert "a_exceeded" in result

    @pytest.mark.parametrize(
        "reading,threshold,expected",
        [
            ({"x": 10.0}, {"x": 5.0}, {"x_exceeded": 1}),
            ({"x": 5.0}, {"x": 5.0}, {"x_exceeded": 0}),
            ({"x": 4.0}, {"x": 5.0}, {"x_exceeded": 0}),
        ],
    )
    def test_parametrized(self, reading, threshold, expected) -> None:
        from app.features import threshold_exceedance_vector

        assert threshold_exceedance_vector(reading, threshold) == expected


class TestRateOfChangeSeries:
    def test_window_1_basic(self) -> None:
        from app.features import rate_of_change

        result = rate_of_change([1.0, 3.0, 6.0], window=1)
        assert result[0] == 0.0
        assert result[1] == pytest.approx(2.0, abs=1e-5)
        assert result[2] == pytest.approx(3.0, abs=1e-5)

    def test_window_2(self) -> None:
        from app.features import rate_of_change

        result = rate_of_change([0.0, 1.0, 3.0, 6.0], window=2)
        assert result[:2] == [0.0, 0.0]
        assert result[2] == pytest.approx(3.0, abs=1e-5)

    def test_invalid_window_raises(self) -> None:
        from app.features import rate_of_change

        with pytest.raises(ValueError):
            rate_of_change([1.0, 2.0], window=0)

    def test_empty_series(self) -> None:
        from app.features import rate_of_change

        assert rate_of_change([], window=1) == []


class TestExponentialDecayFeature:
    def test_same_length_output(self) -> None:
        from app.features import exponential_decay_feature

        values = [1.0, 2.0, 3.0, 4.0]
        result = exponential_decay_feature(values)
        assert len(result) == len(values)

    def test_normalised_to_one(self) -> None:
        from app.features import exponential_decay_feature

        result = exponential_decay_feature([1.0, 2.0, 3.0], decay=0.5)
        assert sum(abs(v) for v in result) == pytest.approx(1.0, abs=1e-4)

    def test_invalid_decay_raises(self) -> None:
        from app.features import exponential_decay_feature

        with pytest.raises(ValueError):
            exponential_decay_feature([1.0, 2.0], decay=0.0)

    def test_all_zeros_returns_zeros(self) -> None:
        from app.features import exponential_decay_feature

        result = exponential_decay_feature([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]


class TestPairwiseRatioFeatures:
    def test_basic_ratio(self) -> None:
        from app.features import pairwise_ratio_features

        result = pairwise_ratio_features({"a": 10.0, "b": 2.0}, [("a", "b")])
        assert result["a_div_b"] == pytest.approx(5.0, abs=1e-5)

    def test_zero_denominator_returns_zero(self) -> None:
        from app.features import pairwise_ratio_features

        result = pairwise_ratio_features({"x": 5.0, "y": 0.0}, [("x", "y")])
        assert result["x_div_y"] == 0.0

    def test_missing_key_defaults_to_zero(self) -> None:
        from app.features import pairwise_ratio_features

        result = pairwise_ratio_features({"a": 4.0}, [("a", "missing")])
        assert result["a_div_missing"] == 0.0

    def test_empty_pairs(self) -> None:
        from app.features import pairwise_ratio_features

        assert pairwise_ratio_features({"a": 1.0}, []) == {}
