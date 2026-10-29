"""Data cleaning and preprocessing utilities for OHLCV time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ohlcv(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error strings for an OHLCV DataFrame.

    Args:
        df: DataFrame with columns open, high, low, close, volume.

    Returns:
        Empty list when valid, otherwise one string per violation.
    """
    required = {"open", "high", "low", "close", "volume"}
    errors: list[str] = []
    missing = required - set(df.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
        return errors
    if (df["high"] < df["low"]).any():
        errors.append("high < low in one or more rows")
    if (df["volume"] < 0).any():
        errors.append("negative volume in one or more rows")
    if df[list(required)].isna().any().any():
        errors.append("NaN values present in OHLCV columns")
    return errors


def fill_missing_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then backward-fill NaN values in OHLCV columns.

    Args:
        df: Input DataFrame (modified in place).

    Returns:
        DataFrame with NaN values filled.
    """
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()
    return df


def normalise_returns(returns: np.ndarray) -> np.ndarray:
    """Z-score normalise a 1D returns array.

    Args:
        returns: 1D array of daily returns.

    Returns:
        Normalised array with mean≈0 and std≈1, or zeros if std is zero.
    """
    std = returns.std()
    if std == 0:
        return np.zeros_like(returns)
    return (returns - returns.mean()) / std


def clip_outliers(arr: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """Winsorise a 1D array at ±z_threshold standard deviations.

    Args:
        arr: Input 1D array.
        z_threshold: Number of standard deviations to use as clip boundary.

    Returns:
        Array with outliers clipped.
    """
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return arr.copy()
    lo = mean - z_threshold * std
    hi = mean + z_threshold * std
    return np.clip(arr, lo, hi)


def compute_log_returns(close_prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a close price series.

    Args:
        close_prices: 1D array of closing prices (length N).

    Returns:
        Log return array of length N-1. Returns empty array for N < 2.
    """
    arr = np.asarray(close_prices, dtype=float)
    if len(arr) < 2:
        return np.empty(0, dtype=float)
    return np.log(arr[1:] / (arr[:-1] + 1e-12))


def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Expand a date column into day-of-week and month integer features.

    Args:
        df: DataFrame containing a parseable date column.
        date_col: Column name holding the date values.

    Returns:
        DataFrame with new columns ``day_of_week`` (0=Mon) and ``month`` (1-12).
        Original ``date_col`` is preserved.
    """
    dates = pd.to_datetime(df[date_col])
    df = df.copy()
    df["day_of_week"] = dates.dt.dayofweek
    df["month"] = dates.dt.month
    return df
