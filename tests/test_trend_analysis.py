"""Tests for app/trend_analysis.py."""

from __future__ import annotations

import pytest

from app.trend_analysis import (
    TrendResult,
    detect_change_points,
    linear_trend,
    percentage_change,
    rolling_mean,
    seasonal_decompose_naive,
)


def test_linear_trend_rising() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = linear_trend(values)
    assert result.direction == "rising"
    assert result.slope > 0


def test_linear_trend_falling() -> None:
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    result = linear_trend(values)
    assert result.direction == "falling"
    assert result.slope < 0


def test_linear_trend_stable() -> None:
    values = [10.0, 10.0, 10.0, 10.0]
    result = linear_trend(values)
    assert result.direction == "stable"


def test_linear_trend_single_value() -> None:
    result = linear_trend([5.0])
    assert isinstance(result, TrendResult)
    assert result.slope == 0.0


def test_linear_trend_perfect_r_squared() -> None:
    values = [2.0 * i for i in range(10)]
    result = linear_trend(values)
    assert result.r_squared == pytest.approx(1.0, abs=1e-4)


def test_percentage_change_increase() -> None:
    assert percentage_change(100.0, 120.0) == pytest.approx(20.0)


def test_percentage_change_decrease() -> None:
    assert percentage_change(100.0, 80.0) == pytest.approx(-20.0)


def test_percentage_change_zero_old() -> None:
    assert percentage_change(0.0, 50.0) == 0.0


def test_rolling_mean_window_1() -> None:
    values = [1.0, 2.0, 3.0]
    result = rolling_mean(values, window=1)
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_rolling_mean_window_2() -> None:
    values = [2.0, 4.0, 6.0, 8.0]
    result = rolling_mean(values, window=2)
    assert result[1] == pytest.approx(3.0)
    assert result[3] == pytest.approx(7.0)


def test_rolling_mean_empty() -> None:
    assert rolling_mean([], window=3) == []


def test_rolling_mean_invalid_window() -> None:
    assert rolling_mean([1.0, 2.0], window=0) == []


def test_detect_change_points_sharp_jump() -> None:
    values = [10.0] * 5 + [100.0] * 5
    cps = detect_change_points(values, threshold=1.5)
    assert 5 in cps


def test_detect_change_points_constant_series() -> None:
    values = [5.0] * 10
    assert detect_change_points(values) == []


def test_detect_change_points_too_short() -> None:
    assert detect_change_points([1.0, 2.0]) == []


def test_seasonal_decompose_returns_keys() -> None:
    values = [float(i % 4) for i in range(16)]
    result = seasonal_decompose_naive(values, period=4)
    assert "trend" in result
    assert "seasonal" in result
    assert "residual" in result


def test_seasonal_decompose_length_matches() -> None:
    values = list(range(24))
    result = seasonal_decompose_naive(values, period=6)
    assert len(result["trend"]) == 24
    assert len(result["seasonal"]) == 24


def test_seasonal_decompose_short_input() -> None:
    values = [1.0, 2.0, 3.0]
    result = seasonal_decompose_naive(values, period=6)
    assert result["trend"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("n,expected_dir", [(5, "rising"), (0, "stable")])
def test_linear_trend_direction_parametrize(n, expected_dir) -> None:
    if n > 0:
        values = list(range(1, n + 1))
    else:
        values = [5.0] * 5
    result = linear_trend(values)
    assert result.direction == expected_dir
