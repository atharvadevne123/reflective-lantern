"""Extended tests for Sports-Oracle time-series forecasting utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.timeseries import forecast_form, linear_trend, sma


def test_sma_empty_returns_zero():
    assert sma([]) == 0.0


def test_sma_single_value():
    assert sma([5.0]) == pytest.approx(5.0)


def test_sma_full_window():
    result = sma([1.0, 2.0, 3.0, 4.0, 5.0], window=5)
    assert result == pytest.approx(3.0)


def test_sma_partial_window_uses_available():
    result = sma([10.0, 20.0], window=5)
    assert result == pytest.approx(15.0)


def test_sma_larger_window_than_list():
    result = sma([1.0, 2.0, 3.0], window=100)
    assert result == pytest.approx(2.0)


def test_sma_uses_last_n_values():
    result = sma([100.0, 1.0, 2.0, 3.0], window=3)
    assert result == pytest.approx(2.0)


def test_linear_trend_empty_returns_zero():
    assert linear_trend([]) == 0.0


def test_linear_trend_single_returns_zero():
    assert linear_trend([5.0]) == 0.0


def test_linear_trend_rising_positive():
    result = linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result > 0


def test_linear_trend_falling_negative():
    result = linear_trend([5.0, 4.0, 3.0, 2.0, 1.0])
    assert result < 0


def test_linear_trend_flat_near_zero():
    result = linear_trend([3.0, 3.0, 3.0, 3.0, 3.0])
    assert result == pytest.approx(0.0, abs=1e-10)


def test_forecast_form_returns_list():
    result = forecast_form([0.4, 0.5, 0.6])
    assert isinstance(result, list)


def test_forecast_form_length_matches_n_steps():
    result = forecast_form([0.4, 0.5, 0.6], n_steps=4)
    assert len(result) == 4


def test_forecast_form_clipped_to_0_1():
    result = forecast_form([0.9, 0.95, 1.0], n_steps=5)
    for v in result:
        assert 0.0 <= v <= 1.0


def test_forecast_form_single_value_uses_sma():
    result = forecast_form([0.7], n_steps=2)
    assert len(result) == 2
    for v in result:
        assert v == pytest.approx(0.7)


def test_forecast_form_rising_trend_continues():
    result = forecast_form([0.3, 0.4, 0.5], n_steps=1)
    assert result[0] > 0.5


@pytest.mark.parametrize("window", [1, 3, 5, 10])
def test_sma_various_windows(window):
    values = [float(i) for i in range(10)]
    result = sma(values, window=window)
    assert isinstance(result, float)
