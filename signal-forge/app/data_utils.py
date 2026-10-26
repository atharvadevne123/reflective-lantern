"""Data utility functions for time-series preprocessing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fill_missing_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then back-fill missing values in OHLCV columns.

    Args:
        df: Raw OHLCV DataFrame that may contain NaN values.

    Returns:
        DataFrame with missing values imputed.
    """
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df.copy()
    df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()
    logger.debug("Filled missing OHLCV values in %d columns", len(ohlcv_cols))
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series.

    Args:
        prices: Series of asset prices.

    Returns:
        Series of log returns (NaN for the first element).
    """
    return np.log(prices / prices.shift(1))


def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a series at specified quantiles to reduce outlier influence.

    Args:
        series: Numeric series to winsorise.
        lower: Lower quantile bound (default 1%).
        upper: Upper quantile bound (default 99%).

    Returns:
        Winsorised series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore_normalise(series: pd.Series) -> pd.Series:
    """Standardise a series to zero mean, unit variance.

    Args:
        series: Numeric series to normalise.

    Returns:
        Z-score normalised series; returns zeros if std is 0.
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def validate_ohlcv_frame(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages for an OHLCV DataFrame.

    Args:
        df: DataFrame to validate.

    Returns:
        List of error strings; empty list means valid.
    """
    errors: list[str] = []
    required = ["close", "volume"]
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if "close" in df.columns and (df["close"] <= 0).any():
        errors.append("close column contains non-positive values")
    if "volume" in df.columns and (df["volume"] < 0).any():
        errors.append("volume column contains negative values")
    return errors
