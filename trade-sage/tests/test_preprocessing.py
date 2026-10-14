"""Tests for OHLCV data cleaning and preprocessing utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    clip_outliers,
    fill_missing_ohlcv,
    normalise_returns,
    validate_ohlcv,
)


def _make_df(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, n).cumsum()
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.integers(1000, 5000, n).astype(float),
    })


def test_validate_clean_df():
    df = _make_df()
    assert validate_ohlcv(df) == []


def test_validate_missing_columns():
    df = _make_df().drop(columns=["volume"])
    errors = validate_ohlcv(df)
    assert any("volume" in e for e in errors)


def test_validate_high_lt_low():
    df = _make_df()
    df.loc[0, "high"] = df.loc[0, "low"] - 1
    errors = validate_ohlcv(df)
    assert any("high < low" in e for e in errors)


def test_validate_negative_volume():
    df = _make_df()
    df.loc[0, "volume"] = -100
    errors = validate_ohlcv(df)
    assert any("negative volume" in e for e in errors)


def test_validate_nan_values():
    df = _make_df()
    df.loc[0, "close"] = float("nan")
    errors = validate_ohlcv(df)
    assert any("NaN" in e for e in errors)


def test_fill_missing_ohlcv():
    df = _make_df()
    df.loc[2, "close"] = float("nan")
    filled = fill_missing_ohlcv(df.copy())
    assert not filled["close"].isna().any()


def test_normalise_returns_mean_near_zero():
    rng = np.random.default_rng(42)
    ret = rng.normal(0.01, 0.02, 200)
    normed = normalise_returns(ret)
    assert abs(normed.mean()) < 0.01


def test_normalise_returns_std_near_one():
    rng = np.random.default_rng(42)
    ret = rng.normal(0.01, 0.02, 200)
    normed = normalise_returns(ret)
    assert abs(normed.std() - 1.0) < 0.01


def test_normalise_constant_returns_zeros():
    ret = np.full(50, 0.005)
    assert (normalise_returns(ret) == 0).all()


def test_clip_outliers_reduces_range():
    arr = np.array([0.0] * 98 + [100.0, -100.0])
    clipped = clip_outliers(arr, z_threshold=2.0)
    assert clipped.max() < 100.0
    assert clipped.min() > -100.0


def test_clip_outliers_constant_array():
    arr = np.full(10, 5.0)
    result = clip_outliers(arr)
    np.testing.assert_array_equal(result, arr)
