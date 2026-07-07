"""Tests for time-series forecasting module."""

import math

import pytest

from app.time_series import (
    compute_sma,
    exponential_smoothing_forecast,
    linear_trend_forecast,
)


def test_sma_length_matches_input():
    result = compute_sma([1, 2, 3, 4, 5], window=3)
    assert len(result) == 5


def test_sma_first_values_nan():
    result = compute_sma([1, 2, 3, 4, 5], window=3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])


def test_sma_correct_value():
    result = compute_sma([100_000, 200_000, 300_000, 400_000], window=3)
    assert result[2] == pytest.approx(200_000.0)
    assert result[3] == pytest.approx(300_000.0)


def test_sma_too_short_all_nan():
    result = compute_sma([100, 200], window=5)
    assert all(math.isnan(v) for v in result)


def test_linear_trend_positive_slope():
    values = [100_000, 110_000, 120_000, 130_000, 140_000]
    result = linear_trend_forecast(values, horizon=2)
    assert result["slope"] > 0
    assert len(result["forecasts"]) == 2
    assert result["r_squared"] > 0.99


def test_linear_trend_flat_r_squared():
    values = [500_000] * 5
    result = linear_trend_forecast(values, horizon=3)
    assert result["slope"] == pytest.approx(0.0, abs=1.0)
    assert result["r_squared"] == 0.0


def test_linear_trend_too_few_values():
    result = linear_trend_forecast([100_000, 120_000], horizon=3)
    assert result["forecasts"] == []
    assert result["r_squared"] == 0.0


def test_linear_trend_forecast_extrapolates():
    values = [100_000 + i * 10_000 for i in range(5)]
    result = linear_trend_forecast(values, horizon=3)
    for f in result["forecasts"]:
        assert f > values[-1]


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.7, 0.9])
def test_exp_smoothing_returns_horizon_values(alpha):
    values = [100_000, 110_000, 120_000, 130_000]
    result = exponential_smoothing_forecast(values, alpha=alpha, horizon=3)
    assert len(result) == 3


def test_exp_smoothing_empty_input():
    assert exponential_smoothing_forecast([], horizon=3) == []


def test_exp_smoothing_single_value():
    result = exponential_smoothing_forecast([500_000], horizon=2)
    assert all(v == pytest.approx(500_000.0) for v in result)
