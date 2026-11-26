"""Extended forecasting tests for sport-cast with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.forecasting import detect_form_trend, linear_trend, rolling_average


def test_linear_trend_empty_returns_zeros() -> None:
    result = linear_trend([])
    assert result["slope"] == 0.0
    assert result["intercept"] == 0.0
    assert result["next_predicted"] == 0.0


def test_linear_trend_single_value() -> None:
    result = linear_trend([5.0])
    assert result["slope"] == 0.0
    assert result["intercept"] == pytest.approx(5.0, rel=1e-4)


def test_linear_trend_two_values_slope() -> None:
    result = linear_trend([0.0, 1.0])
    assert result["slope"] > 0.0


def test_linear_trend_rising_sequence() -> None:
    result = linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["slope"] == pytest.approx(1.0, rel=1e-3)
    assert result["next_predicted"] == pytest.approx(6.0, rel=1e-3)


def test_linear_trend_declining_sequence() -> None:
    result = linear_trend([5.0, 4.0, 3.0, 2.0, 1.0])
    assert result["slope"] < 0


def test_linear_trend_constant_sequence() -> None:
    result = linear_trend([3.0, 3.0, 3.0, 3.0])
    assert result["slope"] == pytest.approx(0.0, abs=1e-6)


def test_linear_trend_has_required_keys() -> None:
    result = linear_trend([1.0, 2.0, 3.0])
    assert "slope" in result
    assert "intercept" in result
    assert "next_predicted" in result


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_rolling_average_length_preserved(n: int) -> None:
    values = list(range(n))
    out = rolling_average([float(v) for v in values], window=3)
    assert len(out) == n


def test_rolling_average_empty() -> None:
    assert rolling_average([]) == []


def test_rolling_average_single_element() -> None:
    out = rolling_average([7.0], window=3)
    assert out == [7.0]


def test_rolling_average_window_one_equals_input() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    out = rolling_average(values, window=1)
    assert out == pytest.approx(values, rel=1e-6)


@pytest.mark.parametrize("window", [2, 3, 5, 10])
def test_rolling_average_non_negative_for_non_negative_input(window: int) -> None:
    values = [float(i) for i in range(20)]
    out = rolling_average(values, window=window)
    assert all(v >= 0 for v in out)


def test_rolling_average_constant_series_equals_constant() -> None:
    values = [5.0] * 10
    out = rolling_average(values, window=3)
    assert out == pytest.approx([5.0] * 10, rel=1e-6)


@pytest.mark.parametrize(
    "win_rates,expected_trend",
    [
        ([0.1, 0.2, 0.3, 0.4, 0.5], "improving"),
        ([0.9, 0.7, 0.5, 0.3, 0.1], "declining"),
        ([0.5, 0.5, 0.5, 0.5, 0.5], "stable"),
        ([0.4, 0.5, 0.4, 0.5, 0.4], "stable"),
    ],
)
def test_detect_form_trend(win_rates: list[float], expected_trend: str) -> None:
    assert detect_form_trend(win_rates) == expected_trend


@pytest.mark.parametrize("n_values", [0, 1, 2])
def test_detect_form_trend_too_few_values_stable(n_values: int) -> None:
    values = [0.5] * n_values
    assert detect_form_trend(values) == "stable"


def test_detect_form_trend_valid_outputs() -> None:
    valid = {"improving", "declining", "stable"}
    for _ in range(10):
        import random

        vals = [random.random() for _ in range(5)]
        result = detect_form_trend(vals)
        assert result in valid


def test_linear_trend_next_predicted_one_step_ahead() -> None:
    values = [0.0, 2.0, 4.0, 6.0]
    result = linear_trend(values)
    assert result["next_predicted"] == pytest.approx(8.0, rel=1e-3)
