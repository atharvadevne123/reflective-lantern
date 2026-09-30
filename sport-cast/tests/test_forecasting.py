"""Forecasting utility tests."""
from __future__ import annotations

import pytest

from app.forecasting import detect_form_trend, linear_trend, rolling_average


def test_linear_trend_increasing():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = linear_trend(values)
    assert result["slope"] == pytest.approx(1.0, abs=0.01)

def test_linear_trend_flat():
    values = [2.0] * 5
    result = linear_trend(values)
    assert abs(result["slope"]) < 0.01

def test_rolling_average_length():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = rolling_average(values, window=3)
    assert len(result) == len(values)

def test_rolling_average_empty():
    assert rolling_average([]) == []

@pytest.mark.parametrize("rates,expected", [
    ([0.2, 0.4, 0.6, 0.8, 1.0], "improving"),
    ([1.0, 0.8, 0.6, 0.4, 0.2], "declining"),
    ([0.5, 0.5, 0.5, 0.5, 0.5], "stable"),
])
def test_detect_form_trend(rates, expected):
    assert detect_form_trend(rates) == expected

def test_detect_form_trend_short():
    assert detect_form_trend([0.5]) == "stable"
