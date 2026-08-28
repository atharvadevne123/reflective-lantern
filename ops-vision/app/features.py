"""Feature engineering pipeline for Ops-Vision SRE incident prediction."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "error_rate_per_min",
    "latency_p99_ms",
    "request_rate_per_sec",
    "disk_io_util_pct",
]


class ResourcePressureTransformer(BaseEstimator, TransformerMixin):
    """Compute composite resource pressure score from CPU and memory."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ResourcePressureTransformer":
        """Fit (no-op — stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add resource_pressure column as weighted sum of CPU and memory."""
        X = X.copy()
        X["resource_pressure"] = (
            0.6 * X["cpu_usage_pct"] + 0.4 * X["memory_usage_pct"]
        ) / 100.0
        logger.debug("ResourcePressureTransformer applied")
        return X


class LatencyErrRatioTransformer(BaseEstimator, TransformerMixin):
    """Compute latency-to-error-rate ratio as a stress signal."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LatencyErrRatioTransformer":
        """Fit (no-op — stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add latency_err_ratio column; clamp denominator to avoid zero-div."""
        X = X.copy()
        denom = X["error_rate_per_min"].clip(lower=0.001)
        X["latency_err_ratio"] = X["latency_p99_ms"] / denom
        logger.debug("LatencyErrRatioTransformer applied")
        return X


class ThroughputPressureTransformer(BaseEstimator, TransformerMixin):
    """Compute normalised throughput pressure from request rate and disk I/O."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ThroughputPressureTransformer":
        """Fit (no-op — stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add throughput_pressure column."""
        X = X.copy()
        X["throughput_pressure"] = (
            X["request_rate_per_sec"] * (1 + X["disk_io_util_pct"] / 100.0)
        )
        logger.debug("ThroughputPressureTransformer applied")
        return X


class LogLatencyTransformer(BaseEstimator, TransformerMixin):
    """Log-transform latency to reduce skewness."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LogLatencyTransformer":
        """Fit (no-op — stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add log_latency_p99 column."""
        X = X.copy()
        X["log_latency_p99"] = np.log1p(X["latency_p99_ms"].clip(lower=0))
        logger.debug("LogLatencyTransformer applied")
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of columns from a DataFrame."""

    def __init__(self, columns: list[str]) -> None:
        """Initialise with the list of columns to keep."""
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnSelector":
        """Fit (no-op — stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return only the specified columns as a numpy array."""
        logger.debug("ColumnSelector selected %d columns", len(self.columns))
        return X[self.columns].values


ENGINEERED_COLS: list[str] = FEATURE_COLS + [
    "resource_pressure",
    "latency_err_ratio",
    "throughput_pressure",
    "log_latency_p99",
]


def build_feature_pipeline() -> Pipeline:
    """Build and return the full sklearn feature engineering pipeline.

    Returns:
        A Pipeline that applies all feature transformers and scaling.
    """
    logger.info("Building feature engineering pipeline")
    pipeline = Pipeline(
        steps=[
            ("resource_pressure", ResourcePressureTransformer()),
            ("latency_err_ratio", LatencyErrRatioTransformer()),
            ("throughput_pressure", ThroughputPressureTransformer()),
            ("log_latency", LogLatencyTransformer()),
            ("selector", ColumnSelector(ENGINEERED_COLS)),
            ("scaler", RobustScaler()),
        ]
    )
    logger.info("Feature pipeline built with %d steps", len(pipeline.steps))
    return pipeline


def dataframe_from_dict(payload: dict) -> pd.DataFrame:
    """Convert a flat metric dict to a one-row DataFrame for prediction.

    Args:
        payload: Dict with metric key-value pairs.

    Returns:
        Single-row DataFrame with FEATURE_COLS as columns.
    """
    row = {col: payload.get(col, 0.0) for col in FEATURE_COLS}
    return pd.DataFrame([row])
