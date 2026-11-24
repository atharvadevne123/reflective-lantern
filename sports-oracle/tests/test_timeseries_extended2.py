"""Extended time-series utility tests — second pass with parametrize."""
from __future__ import annotations

import pytest

from app.timeseries import forecast_form, linear_trend, sma


def test_sma_empty_returns_zero() -> None:
    assert sma([]) == pytest.approx(0.0)


def test_sma_single_value() -> None:
    assert sma([7.0]) == pytest.approx(7.0)


def test_sma_window_larger_than_series_uses_all() -> None:
    values = [2.0, 4.0, 6.0]
    assert sma(values, window=10) == pytest.approx(4.0)


def test_sma_window_one_returns_last() -> None:
    values = [1.0, 2.0, 3.0, 9.0]
    assert sma(values, window=1) == pytest.approx(9.0)


@pytest.mark.parametrize("window,expected", [
    (1, 5.0),
    (2, 4.5),
    (3, 4.0),
    (5, 3.0),
])
def test_sma_parametrized_windows(window: int, expected: float) -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sma(values, window=window) == pytest.approx(expected)


def test_linear_trend_too_short_returns_zero() -> None:
    assert linear_trend([5.0]) == pytest.approx(0.0)


def test_linear_trend_rising_series_positive() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert linear_trend(values) > 0


def test_linear_trend_falling_series_negative() -> None:
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert linear_trend(values) < 0


def test_linear_trend_flat_series_near_zero() -> None:
    values = [5.0] * 10
    assert linear_trend(values) == pytest.approx(0.0, abs=1e-6)


def test_linear_trend_two_points() -> None:
    values = [0.0, 10.0]
    assert linear_trend(values) == pytest.approx(10.0)


def test_forecast_form_too_short_repeats_sma() -> None:
    result = forecast_form([0.5], n_steps=3)
    assert len(result) == 3
    assert all(v == pytest.approx(0.5) for v in result)


def test_forecast_form_length_matches_n_steps() -> None:
    historical = [0.4, 0.5, 0.6, 0.7]
    result = forecast_form(historical, n_steps=5)
    assert len(result) == 5


def test_forecast_form_values_clipped_to_unit_interval() -> None:
    historical = [0.9] * 10
    result = forecast_form(historical, n_steps=10)
    assert all(0.0 <= v <= 1.0 for v in result)


@pytest.mark.parametrize("n_steps", [1, 3, 5, 10])
def test_forecast_form_various_n_steps(n_steps: int) -> None:
    historical = [0.3, 0.4, 0.5, 0.6, 0.7]
    result = forecast_form(historical, n_steps=n_steps)
    assert len(result) == n_steps


def test_forecast_form_flat_signal_stable() -> None:
    historical = [0.5] * 20
    result = forecast_form(historical, n_steps=5)
    for v in result:
        assert v == pytest.approx(0.5, abs=0.01)
