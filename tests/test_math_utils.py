"""Tests for app.math_utils module."""

from __future__ import annotations

import pytest

from app.math_utils import (
    clamp,
    is_close,
    linear_interpolate,
    moving_average,
    percent_change,
    round_to_decimals,
    safe_divide,
)


class TestClamp:
    def test_value_below_low(self) -> None:
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_value_above_high(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_value_within_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_value_at_low_boundary(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_value_at_high_boundary(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == 10.0

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError):
            clamp(5.0, 10.0, 0.0)

    @pytest.mark.parametrize(
        ("value", "low", "high", "expected"),
        [
            (-100.0, 0.0, 1.0, 0.0),
            (0.5, 0.0, 1.0, 0.5),
            (100.0, 0.0, 1.0, 1.0),
        ],
    )
    def test_parametrized(self, value: float, low: float, high: float, expected: float) -> None:
        assert clamp(value, low, high) == expected


class TestSafeDivide:
    def test_normal_division(self) -> None:
        assert safe_divide(10.0, 2.0) == 5.0

    def test_division_by_zero_returns_default(self) -> None:
        assert safe_divide(5.0, 0.0) == 0.0

    def test_custom_default(self) -> None:
        assert safe_divide(5.0, 0.0, default=-1.0) == -1.0

    def test_negative_numerator(self) -> None:
        assert safe_divide(-6.0, 3.0) == -2.0

    @pytest.mark.parametrize(
        ("num", "den", "expected"),
        [(10, 5, 2.0), (0, 5, 0.0), (7, 2, 3.5)],
    )
    def test_parametrized(self, num: float, den: float, expected: float) -> None:
        assert safe_divide(num, den) == expected


class TestPercentChange:
    def test_increase(self) -> None:
        assert percent_change(100.0, 110.0) == pytest.approx(10.0)

    def test_decrease(self) -> None:
        assert percent_change(100.0, 90.0) == pytest.approx(-10.0)

    def test_no_change(self) -> None:
        assert percent_change(50.0, 50.0) == pytest.approx(0.0)

    def test_zero_old_returns_zero(self) -> None:
        assert percent_change(0.0, 100.0) == 0.0

    def test_negative_baseline(self) -> None:
        result = percent_change(-50.0, -25.0)
        assert isinstance(result, float)


class TestRoundToDecimals:
    def test_default_two_decimals(self) -> None:
        assert round_to_decimals(3.14159) == 3.14

    def test_zero_decimals(self) -> None:
        assert round_to_decimals(3.7, 0) == 4.0

    def test_three_decimals(self) -> None:
        assert round_to_decimals(1.23456, 3) == 1.235

    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [(2.555, 2, 2.55), (1.005, 2, 1.0), (9.999, 1, 10.0)],
    )
    def test_parametrized(self, value: float, decimals: int, expected: float) -> None:
        result = round_to_decimals(value, decimals)
        assert isinstance(result, float)


class TestLinearInterpolate:
    def test_midpoint(self) -> None:
        assert linear_interpolate(0.0, 0.0, 10.0, 10.0, 5.0) == pytest.approx(5.0)

    def test_at_x0(self) -> None:
        assert linear_interpolate(0.0, 5.0, 10.0, 15.0, 0.0) == pytest.approx(5.0)

    def test_at_x1(self) -> None:
        assert linear_interpolate(0.0, 5.0, 10.0, 15.0, 10.0) == pytest.approx(15.0)

    def test_equal_x_returns_y0(self) -> None:
        assert linear_interpolate(3.0, 7.0, 3.0, 11.0, 3.0) == 7.0

    def test_extrapolation(self) -> None:
        result = linear_interpolate(0.0, 0.0, 10.0, 10.0, 15.0)
        assert result == pytest.approx(15.0)


class TestMovingAverage:
    def test_window_one_returns_same_values(self) -> None:
        vals = [1.0, 2.0, 3.0]
        assert moving_average(vals, 1) == pytest.approx(vals)

    def test_window_full_size(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average(vals, len(vals))
        assert result[-1] == pytest.approx(3.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            moving_average([], 3)

    def test_window_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            moving_average([1.0, 2.0], 0)

    def test_output_length_matches_input(self) -> None:
        vals = [float(i) for i in range(10)]
        assert len(moving_average(vals, 3)) == len(vals)

    def test_window_two(self) -> None:
        vals = [2.0, 4.0, 6.0, 8.0]
        result = moving_average(vals, 2)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)


class TestIsClose:
    def test_identical_values(self) -> None:
        assert is_close(1.0, 1.0) is True

    def test_close_values(self) -> None:
        assert is_close(1.0, 1.0 + 1e-10) is True

    def test_far_values(self) -> None:
        assert is_close(1.0, 2.0) is False

    def test_abs_tol(self) -> None:
        assert is_close(0.0, 0.0001, abs_tol=0.001) is True

    def test_nan_not_close(self) -> None:
        assert is_close(float("nan"), float("nan")) is False
