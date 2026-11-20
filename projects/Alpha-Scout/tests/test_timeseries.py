"""Time-series forecasting tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.timeseries import compute_exponential_smoothing, forecast_next_return, rolling_z_score


@pytest.fixture
def price_series():
    rng = np.random.default_rng(42)
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, 100))
    return pd.Series(prices)


def test_exponential_smoothing_length(price_series):
    smoothed = compute_exponential_smoothing(price_series)
    assert len(smoothed) == len(price_series)


def test_exponential_smoothing_first_value(price_series):
    smoothed = compute_exponential_smoothing(price_series)
    assert smoothed.iloc[0] == price_series.iloc[0]


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.9])
def test_exponential_smoothing_various_alphas(price_series, alpha):
    smoothed = compute_exponential_smoothing(price_series, alpha=alpha)
    assert len(smoothed) == len(price_series)
    assert not np.isnan(smoothed).any()


def test_rolling_z_score_length(price_series):
    z = rolling_z_score(price_series, window=10)
    assert len(z) == len(price_series)


def test_forecast_returns_dict(price_series):
    result = forecast_next_return(price_series, horizon=5)
    assert "forecast_return" in result
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert result["lower_bound"] <= result["upper_bound"]


def test_forecast_short_series():
    short = pd.Series([100.0, 101.0, 102.0])
    result = forecast_next_return(short, horizon=5)
    assert result["forecast_return"] == 0.0
