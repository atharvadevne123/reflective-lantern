"""Feature engineering pipeline for multivariate time-series data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

WINDOW_SIZES = [5, 10, 20]
LAG_STEPS = [1, 2, 3, 5]


def add_rolling_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add rolling mean, std, min, max for each window size."""
    result = df.copy()
    for col in columns:
        for w in WINDOW_SIZES:
            result[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            result[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
            result[f"{col}_roll_min_{w}"] = df[col].rolling(w, min_periods=1).min()
            result[f"{col}_roll_max_{w}"] = df[col].rolling(w, min_periods=1).max()
    return result


def add_lag_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add lag features for each step."""
    result = df.copy()
    for col in columns:
        for lag in LAG_STEPS:
            result[f"{col}_lag_{lag}"] = df[col].shift(lag).bfill().fillna(0)
    return result


def add_rate_of_change(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add first-order difference (rate of change) features."""
    result = df.copy()
    for col in columns:
        result[f"{col}_roc"] = df[col].diff().fillna(0)
        result[f"{col}_roc2"] = df[col].diff().diff().fillna(0)
    return result


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclical time encoding from a timestamp column."""
    result = df.copy()
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col])
        result["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        result["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        result["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    return result


def add_cross_sensor_correlation(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add pairwise rolling correlations between sensor channels."""
    result = df.copy()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            key = f"corr_{col_a}_{col_b}"
            result[key] = df[col_a].rolling(10, min_periods=2).corr(df[col_b]).fillna(0)
    return result


def build_feature_matrix(
    readings: list[dict[str, Any]],
    value_keys: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert raw readings list into a full feature matrix.

    Returns the feature dataframe and the list of feature column names.
    """
    df = pd.DataFrame(readings)
    if value_keys is None:
        value_keys = [c for c in df.columns if c not in ("timestamp", "sensor_id")]

    df = add_rolling_statistics(df, value_keys)
    df = add_lag_features(df, value_keys)
    df = add_rate_of_change(df, value_keys)
    df = add_time_features(df)
    if len(value_keys) > 1:
        df = add_cross_sensor_correlation(df, value_keys)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "sensor_id")]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols


def build_sklearn_pipeline() -> Pipeline:
    """Return a reusable sklearn preprocessing pipeline."""
    return Pipeline([("scaler", RobustScaler())])
