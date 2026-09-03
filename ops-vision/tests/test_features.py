"""Tests for Ops-Vision feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from app.features import (
    ENGINEERED_COLS,
    FEATURE_COLS,
    ColumnSelector,
    LatencyErrRatioTransformer,
    LogLatencyTransformer,
    ResourcePressureTransformer,
    ThroughputPressureTransformer,
    build_feature_pipeline,
    dataframe_from_dict,
)


class TestResourcePressureTransformer:
    """Tests for ResourcePressureTransformer."""

    def make_df(self, cpu: float, mem: float) -> pd.DataFrame:
        """Create a minimal DataFrame for pressure tests."""
        return pd.DataFrame([{"cpu_usage_pct": cpu, "memory_usage_pct": mem}])

    def test_adds_resource_pressure_column(self):
        """Transformer adds resource_pressure column."""
        t = ResourcePressureTransformer()
        df = self.make_df(60.0, 40.0)
        out = t.fit_transform(df)
        assert "resource_pressure" in out.columns

    def test_pressure_range(self):
        """resource_pressure is between 0 and 1 for valid inputs."""
        t = ResourcePressureTransformer()
        df = self.make_df(80.0, 80.0)
        out = t.fit_transform(df)
        assert 0.0 <= out["resource_pressure"].iloc[0] <= 1.0

    @pytest.mark.parametrize(
        "cpu,mem,expected",
        [
            (100.0, 100.0, 1.0),
            (0.0, 0.0, 0.0),
            (60.0, 0.0, 0.36),
        ],
    )
    def test_pressure_calculation(self, cpu, mem, expected):
        """resource_pressure matches the 0.6*cpu + 0.4*mem / 100 formula."""
        t = ResourcePressureTransformer()
        df = self.make_df(cpu, mem)
        out = t.fit_transform(df)
        assert abs(out["resource_pressure"].iloc[0] - expected) < 1e-6


class TestLatencyErrRatioTransformer:
    """Tests for LatencyErrRatioTransformer."""

    def make_df(self, latency: float, error_rate: float) -> pd.DataFrame:
        return pd.DataFrame([{"latency_p99_ms": latency, "error_rate_per_min": error_rate}])

    def test_adds_latency_err_ratio(self):
        """Transformer adds latency_err_ratio column."""
        t = LatencyErrRatioTransformer()
        df = self.make_df(500.0, 10.0)
        out = t.fit_transform(df)
        assert "latency_err_ratio" in out.columns

    def test_zero_error_rate_no_division_error(self):
        """Zero error rate is clamped to 0.001, preventing ZeroDivisionError."""
        t = LatencyErrRatioTransformer()
        df = self.make_df(500.0, 0.0)
        out = t.fit_transform(df)
        assert np.isfinite(out["latency_err_ratio"].iloc[0])

    def test_ratio_correct(self):
        """Ratio should equal latency / error_rate."""
        t = LatencyErrRatioTransformer()
        df = self.make_df(500.0, 5.0)
        out = t.fit_transform(df)
        assert abs(out["latency_err_ratio"].iloc[0] - 100.0) < 1e-4


class TestLogLatencyTransformer:
    """Tests for LogLatencyTransformer."""

    def test_adds_log_latency_column(self):
        """Transformer adds log_latency_p99 column."""
        t = LogLatencyTransformer()
        df = pd.DataFrame([{"latency_p99_ms": 200.0}])
        out = t.fit_transform(df)
        assert "log_latency_p99" in out.columns

    def test_log_latency_is_finite_for_zero(self):
        """log1p(0) = 0, not -inf."""
        t = LogLatencyTransformer()
        df = pd.DataFrame([{"latency_p99_ms": 0.0}])
        out = t.fit_transform(df)
        assert out["log_latency_p99"].iloc[0] == pytest.approx(0.0)


class TestThroughputPressureTransformer:
    """Tests for ThroughputPressureTransformer."""

    def test_adds_throughput_pressure(self):
        """Transformer adds throughput_pressure column."""
        t = ThroughputPressureTransformer()
        df = pd.DataFrame([{"request_rate_per_sec": 200.0, "disk_io_util_pct": 50.0}])
        out = t.fit_transform(df)
        assert "throughput_pressure" in out.columns

    def test_throughput_value_correct(self):
        """throughput_pressure = request_rate * (1 + disk_io/100)."""
        t = ThroughputPressureTransformer()
        df = pd.DataFrame([{"request_rate_per_sec": 100.0, "disk_io_util_pct": 0.0}])
        out = t.fit_transform(df)
        assert out["throughput_pressure"].iloc[0] == pytest.approx(100.0)


class TestColumnSelector:
    """Tests for ColumnSelector."""

    def test_selects_correct_columns(self, synthetic_dataframe):
        """ColumnSelector returns only the specified columns as array."""
        cols = ["cpu_usage_pct", "memory_usage_pct"]
        selector = ColumnSelector(cols)
        result = selector.fit_transform(synthetic_dataframe)
        assert result.shape[1] == 2

    def test_output_is_numpy_array(self, synthetic_dataframe):
        """ColumnSelector output type is np.ndarray."""
        selector = ColumnSelector(["cpu_usage_pct"])
        result = selector.fit_transform(synthetic_dataframe)
        assert isinstance(result, np.ndarray)


class TestBuildFeaturePipeline:
    """Tests for the full feature pipeline."""

    def test_pipeline_fit_transform(self, synthetic_dataframe):
        """Pipeline fit_transform produces a numeric array."""
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(synthetic_dataframe)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(synthetic_dataframe)

    def test_pipeline_output_is_finite(self, synthetic_dataframe):
        """All pipeline output values are finite."""
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(synthetic_dataframe)
        assert np.isfinite(X).all()

    def test_pipeline_output_columns(self, synthetic_dataframe):
        """Pipeline output has the expected number of feature columns."""
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(synthetic_dataframe)
        assert X.shape[1] == len(ENGINEERED_COLS)


class TestDataframeFromDict:
    """Tests for the dataframe_from_dict helper."""

    def test_returns_one_row_dataframe(self):
        """Returns a DataFrame with one row."""
        payload = {col: 50.0 for col in FEATURE_COLS}
        df = dataframe_from_dict(payload)
        assert len(df) == 1

    def test_missing_keys_default_to_zero(self):
        """Missing metric keys default to 0.0."""
        df = dataframe_from_dict({"service_name": "svc"})
        assert df["cpu_usage_pct"].iloc[0] == 0.0

    @pytest.mark.parametrize("col", FEATURE_COLS)
    def test_all_feature_cols_present(self, col):
        """All FEATURE_COLS should appear in the output DataFrame."""
        payload = {c: 42.0 for c in FEATURE_COLS}
        df = dataframe_from_dict(payload)
        assert col in df.columns
