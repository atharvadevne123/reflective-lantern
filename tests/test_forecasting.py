"""Tests for app/forecasting.py."""

from __future__ import annotations

import pytest

from app.forecasting import (
    drift_forecast,
    exponential_smoothing_forecast,
    forecast_summary,
    naive_forecast,
    seasonal_naive_forecast,
)

HISTORY = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0]


def test_naive_forecast_length():
    result = naive_forecast(10.0, steps=5)
    assert len(result) == 5


def test_naive_forecast_constant():
    result = naive_forecast(7.5, steps=3)
    assert all(v == pytest.approx(7.5) for v in result)


def test_naive_forecast_zero_steps():
    assert naive_forecast(10.0, steps=0) == []


def test_naive_forecast_negative_steps():
    assert naive_forecast(10.0, steps=-1) == []


def test_drift_forecast_length():
    result = drift_forecast(HISTORY, steps=4)
    assert len(result) == 4


def test_drift_forecast_rising_history():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = drift_forecast(values, steps=2)
    assert result[1] > result[0]


def test_drift_forecast_single_value():
    result = drift_forecast([10.0], steps=3)
    assert len(result) == 3
    assert all(v == pytest.approx(10.0) for v in result)


def test_drift_forecast_empty():
    assert drift_forecast([], steps=3) == []


def test_seasonal_naive_forecast_length():
    result = seasonal_naive_forecast(HISTORY, steps=3, period=3)
    assert len(result) == 3


def test_seasonal_naive_forecast_repeats_pattern():
    values = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    result = seasonal_naive_forecast(values, steps=3, period=3)
    assert len(result) == 3


def test_seasonal_naive_forecast_empty():
    assert seasonal_naive_forecast([], steps=3) == []


def test_exponential_smoothing_forecast_length():
    result = exponential_smoothing_forecast(HISTORY, steps=5)
    assert len(result) == 5


def test_exponential_smoothing_forecast_all_equal():
    result = exponential_smoothing_forecast(HISTORY, steps=3, alpha=0.3)
    assert result[0] == result[1] == result[2]


def test_exponential_smoothing_invalid_alpha():
    with pytest.raises(ValueError):
        exponential_smoothing_forecast(HISTORY, steps=3, alpha=0.0)


def test_exponential_smoothing_alpha_1():
    result = exponential_smoothing_forecast([10.0, 20.0, 30.0], steps=2, alpha=1.0)
    assert all(v == pytest.approx(30.0) for v in result)


def test_forecast_summary_correct():
    forecasts = [10.0, 20.0, 30.0]
    s = forecast_summary(forecasts)
    assert s["mean"] == pytest.approx(20.0)
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    assert s["total"] == pytest.approx(60.0)
    assert s["steps"] == 3


def test_forecast_summary_empty():
    s = forecast_summary([])
    assert s["steps"] == 0
    assert s["mean"] == 0.0


@pytest.mark.parametrize("steps", [1, 3, 6, 24])
def test_naive_forecast_various_steps(steps):
    result = naive_forecast(15.0, steps=steps)
    assert len(result) == steps
