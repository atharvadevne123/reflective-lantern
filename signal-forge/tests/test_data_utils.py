"""Tests for data utility functions."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from app.data_utils import (
    compute_log_returns,
    fill_missing_ohlcv,
    validate_ohlcv_frame,
    winsorise,
    zscore_normalise,
)


def test_fill_missing_ohlcv_forward_fills():
    df = pd.DataFrame({"close": [100.0, np.nan, 102.0], "volume": [1000.0, np.nan, 1200.0]})
    result = fill_missing_ohlcv(df)
    assert result["close"].iloc[1] == 100.0
    assert result["volume"].iloc[1] == 1000.0


def test_compute_log_returns_first_is_nan():
    prices = pd.Series([100.0, 105.0, 103.0])
    returns = compute_log_returns(prices)
    assert np.isnan(returns.iloc[0])
    assert not np.isnan(returns.iloc[1])


def test_compute_log_returns_value():
    prices = pd.Series([100.0, 110.0])
    returns = compute_log_returns(prices)
    assert returns.iloc[1] == pytest.approx(np.log(110.0 / 100.0))


def test_winsorise_clips_outliers():
    s = pd.Series(range(100))
    result = winsorise(s, lower=0.05, upper=0.95)
    assert result.min() >= s.quantile(0.05) - 1
    assert result.max() <= s.quantile(0.95) + 1


def test_zscore_normalise_mean_zero():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore_normalise(s)
    assert abs(result.mean()) < 1e-10


def test_zscore_normalise_constant_series():
    s = pd.Series([5.0] * 10)
    result = zscore_normalise(s)
    assert (result == 0.0).all()


def test_validate_ohlcv_frame_valid(sample_ohlcv_df):
    errors = validate_ohlcv_frame(sample_ohlcv_df)
    assert errors == []


def test_validate_ohlcv_frame_missing_close():
    df = pd.DataFrame({"volume": [1000.0]})
    errors = validate_ohlcv_frame(df)
    assert any("close" in e for e in errors)


@pytest.mark.parametrize("col", ["close", "volume"])
def test_validate_ohlcv_frame_required_columns(col):
    df = pd.DataFrame({"close": [100.0], "volume": [1000.0]}).drop(columns=[col])
    errors = validate_ohlcv_frame(df)
    assert len(errors) > 0
