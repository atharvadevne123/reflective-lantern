"""Time-series forecasting tests for Sports-Oracle."""

from __future__ import annotations

import pytest


def test_sma_basic():
    from app.timeseries import sma

    assert abs(sma([0.4, 0.5, 0.6, 0.7, 0.8], window=3) - 0.7) < 1e-6


def test_sma_empty():
    from app.timeseries import sma

    assert sma([], window=5) == 0.0


def test_sma_window_larger_than_values():
    from app.timeseries import sma

    assert abs(sma([0.5, 0.6], window=10) - 0.55) < 1e-6


def test_linear_trend_rising():
    from app.timeseries import linear_trend

    assert linear_trend([0.3, 0.4, 0.5, 0.6]) > 0


def test_linear_trend_falling():
    from app.timeseries import linear_trend

    assert linear_trend([0.8, 0.6, 0.4, 0.2]) < 0


def test_linear_trend_flat():
    from app.timeseries import linear_trend

    assert abs(linear_trend([0.5, 0.5, 0.5])) < 1e-8


def test_linear_trend_single_value():
    from app.timeseries import linear_trend

    assert linear_trend([0.5]) == 0.0


def test_forecast_form_clipped_to_01():
    from app.timeseries import forecast_form

    # Strongly rising form should clip at 1.0
    forecast = forecast_form([0.8, 0.9, 1.0], n_steps=3)
    assert all(0.0 <= f <= 1.0 for f in forecast)


@pytest.mark.parametrize("n_steps", [1, 3, 5])
def test_forecast_form_returns_correct_length(n_steps):
    from app.timeseries import forecast_form

    result = forecast_form([0.5, 0.55, 0.6], n_steps=n_steps)
    assert len(result) == n_steps
