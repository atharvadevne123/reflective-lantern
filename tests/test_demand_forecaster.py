"""Tests for app.demand_forecaster module."""

from __future__ import annotations

import math

import pytest

from app.demand_forecaster import (
    ForecastMethod,
    ForecastResult,
    exponential_smoothing_forecast,
    forecast_demand,
    linear_trend_forecast,
    mean_absolute_error,
    moving_average_forecast,
    naive_forecast,
    root_mean_square_error,
)


class TestNaiveForecast:
    def test_repeats_last_value(self) -> None:
        result = naive_forecast([10.0, 20.0, 30.0], 3)
        assert all(v == pytest.approx(30.0) for v in result)

    def test_correct_length(self) -> None:
        result = naive_forecast([5.0, 6.0], 5)
        assert len(result) == 5

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            naive_forecast([], 3)

    def test_zero_horizon_raises(self) -> None:
        with pytest.raises(ValueError):
            naive_forecast([10.0], 0)

    @pytest.mark.parametrize("horizon", [1, 5, 10])
    def test_horizon_length(self, horizon: int) -> None:
        assert len(naive_forecast([1.0, 2.0], horizon)) == horizon


class TestMovingAverageForecast:
    def test_window_2(self) -> None:
        result = moving_average_forecast([10.0, 20.0, 30.0], 2, 1)
        assert result[0] == pytest.approx(25.0)

    def test_window_exceeds_history_raises(self) -> None:
        with pytest.raises(ValueError):
            moving_average_forecast([10.0, 20.0], 5, 3)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError):
            moving_average_forecast([10.0], 0, 1)

    def test_correct_length(self) -> None:
        result = moving_average_forecast([1.0, 2.0, 3.0, 4.0, 5.0], 3, 4)
        assert len(result) == 4

    @pytest.mark.parametrize("w", [1, 2, 3])
    def test_non_negative_result(self, w: int) -> None:
        hist = [10.0, 15.0, 20.0, 25.0, 30.0]
        result = moving_average_forecast(hist, w, 3)
        assert all(v > 0 for v in result)


class TestExponentialSmoothingForecast:
    def test_constant_series(self) -> None:
        result = exponential_smoothing_forecast([5.0] * 10, 0.3, 3)
        assert all(v == pytest.approx(5.0) for v in result)

    def test_zero_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            exponential_smoothing_forecast([1.0, 2.0], 0.0, 3)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            exponential_smoothing_forecast([], 0.3, 3)

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_valid_alphas(self, alpha: float) -> None:
        result = exponential_smoothing_forecast([10.0, 12.0, 11.0], alpha, 2)
        assert len(result) == 2


class TestLinearTrendForecast:
    def test_upward_trend(self) -> None:
        hist = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = linear_trend_forecast(hist, 1)
        assert result[0] > 5.0

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValueError):
            linear_trend_forecast([10.0], 3)

    def test_correct_length(self) -> None:
        result = linear_trend_forecast([1.0, 2.0, 3.0], 5)
        assert len(result) == 5

    @pytest.mark.parametrize("horizon", [1, 3, 6])
    def test_horizon_length(self, horizon: int) -> None:
        assert len(linear_trend_forecast([1.0, 2.0, 3.0, 4.0], horizon)) == horizon


class TestMeanAbsoluteError:
    def test_perfect_forecast_zero(self) -> None:
        assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_basic(self) -> None:
        assert mean_absolute_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx(2.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_absolute_error([1.0, 2.0], [1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            mean_absolute_error([], [])

    @pytest.mark.parametrize("n", [2, 5, 10])
    def test_non_negative(self, n: int) -> None:
        actual = [float(i) for i in range(n)]
        pred = [float(i) + 1.0 for i in range(n)]
        assert mean_absolute_error(actual, pred) >= 0.0


class TestRootMeanSquareError:
    def test_perfect_forecast_zero(self) -> None:
        assert root_mean_square_error([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        assert root_mean_square_error([0.0, 0.0], [3.0, 4.0]) >= 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            root_mean_square_error([1.0], [1.0, 2.0])


class TestForecastDemand:
    def test_returns_result(self) -> None:
        result = forecast_demand([10.0, 12.0, 11.0, 13.0], 3)
        assert isinstance(result, ForecastResult)

    def test_correct_horizon(self) -> None:
        result = forecast_demand([10.0] * 10, 5)
        assert result.horizon == 5
        assert len(result.forecast_values) == 5

    @pytest.mark.parametrize("method", list(ForecastMethod))
    def test_all_methods(self, method: ForecastMethod) -> None:
        hist = [float(i) for i in range(1, 11)]
        result = forecast_demand(hist, 3, method=method)
        assert len(result.forecast_values) == 3
