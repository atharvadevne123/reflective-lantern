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
