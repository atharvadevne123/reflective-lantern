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


def sensor_zscore_row(row: dict[str, float]) -> dict[str, float]:
    """Compute z-scores for each sensor value in a single reading row.

    Args:
        row: Dict mapping sensor name → float value.

    Returns:
        Dict mapping ``<sensor>_zscore`` → standardised value;
        all zeros when all values are equal.
    """
    if not row:
        return {}
    values = list(row.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5
    if std == 0.0:
        return {f"{k}_zscore": 0.0 for k in row}
    return {f"{k}_zscore": round((v - mean) / std, 6) for k, v in row.items()}


def threshold_exceedance_vector(
    row: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, int]:
    """Return binary flags indicating which sensors exceed their thresholds.

    Args:
        row: Current sensor readings.
        thresholds: Per-sensor upper limits.

    Returns:
        Dict mapping ``<sensor>_exceeded`` → 1 if exceeded, 0 otherwise.
        Sensors absent from *thresholds* are skipped.
    """
    return {f"{k}_exceeded": int(row[k] > thresholds[k]) for k in row if k in thresholds}


def rate_of_change(
    values: list[float],
    window: int = 1,
) -> list[float]:
    """Compute the rate of change over *window* time steps.

    Args:
        values: Input numeric series.
        window: Number of steps over which to compute the change. Default 1.

    Returns:
        List of changes (same length as *values*); the first *window* entries
        are 0.0 (no prior history).

    Raises:
        ValueError: If *window* < 1.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    result = [0.0] * window
    for i in range(window, len(values)):
        result.append(round(values[i] - values[i - window], 6))
    return result


def exponential_decay_feature(
    values: list[float],
    decay: float = 0.9,
) -> list[float]:
    """Apply exponential decay weighting to a series.

    Each value is weighted by *decay^(n-1-i)* relative to the most recent
    observation, then the series is normalised to sum to 1 if non-zero.

    Args:
        values: Input numeric series.
        decay: Decay factor in (0, 1]. Lower values weight recent obs more heavily.

    Returns:
        Decay-weighted series the same length as *values*.

    Raises:
        ValueError: If *decay* is outside (0, 1].
    """
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    n = len(values)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    weighted = [v * w for v, w in zip(values, weights, strict=False)]
    total = sum(abs(w) for w in weighted)
    if total < 1e-12:
        return [0.0] * n
    return [round(w / total, 6) for w in weighted]


def pairwise_ratio_features(
    row: dict[str, float],
    pairs: list[tuple[str, str]],
) -> dict[str, float]:
    """Compute ratio features for specified sensor pairs.

    Args:
        row: Current sensor readings.
        pairs: List of (numerator_key, denominator_key) tuples.

    Returns:
        Dict mapping ``<a>_div_<b>`` → ratio value.
        Pairs where the denominator is 0 return 0.0.
    """
    result: dict[str, float] = {}
    for a, b in pairs:
        val_a = row.get(a, 0.0)
        val_b = row.get(b, 0.0)
        result[f"{a}_div_{b}"] = round(val_a / val_b, 6) if val_b != 0.0 else 0.0
    return result
