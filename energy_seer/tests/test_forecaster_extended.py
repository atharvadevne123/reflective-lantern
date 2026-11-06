"""Extended tests for energy_seer app/forecaster.py."""

from __future__ import annotations

import pytest

from app.forecaster import (
    exponential_smoothing,
    forecast_confidence_band,
    forecast_horizon_accuracy,
    linear_trend,
    moving_average_forecast,
    weighted_moving_average,
)


class TestLinearTrendExtended:
    def test_decreasing_slope_negative(self) -> None:
        values = [10.0, 8.0, 6.0, 4.0, 2.0]
        result = linear_trend(values)
        assert result["slope"] < 0

    def test_empty_list_returns_zero_slope(self) -> None:
        result = linear_trend([])
        assert result["slope"] == 0.0

    def test_two_values_slope_correct(self) -> None:
        result = linear_trend([1.0, 3.0])
        assert result["slope"] == pytest.approx(2.0, abs=0.01)

    def test_next_extrapolates_beyond_last(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = linear_trend(values)
        assert result["next"] > values[-1]

    def test_returns_required_keys(self) -> None:
        result = linear_trend([1.0, 2.0])
        for key in ("slope", "intercept", "next"):
            assert key in result


class TestMovingAverageForecastExtended:
    def test_forecast_length_matches_steps(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average_forecast(values, window=3, steps=4)
        assert len(result) == 4

    def test_flat_series_forecast_equals_value(self) -> None:
        values = [5.0] * 10
        result = moving_average_forecast(values, window=5, steps=1)
        assert result[0] == pytest.approx(5.0)

    def test_window_larger_than_series(self) -> None:
        values = [3.0, 4.0]
        result = moving_average_forecast(values, window=100, steps=1)
        assert len(result) == 1


class TestExponentialSmoothingExtended:
    def test_flat_series_returns_same_value(self) -> None:
        values = [7.0] * 5
        result = exponential_smoothing(values, alpha=0.3, steps=2)
        assert all(r == pytest.approx(7.0) for r in result)

    def test_steps_matches_output_length(self) -> None:
        result = exponential_smoothing([1.0, 2.0, 3.0], alpha=0.5, steps=3)
        assert len(result) == 3

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            exponential_smoothing([])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            exponential_smoothing([1.0, 2.0], alpha=0.0)

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
    def test_valid_alpha_accepted(self, alpha: float) -> None:
        result = exponential_smoothing([1.0, 2.0, 3.0], alpha=alpha, steps=1)
        assert len(result) == 1


class TestForecastHorizonAccuracyExtended:
    def test_perfect_forecast_zero_error(self) -> None:
        vals = [1.0, 2.0, 3.0]
        result = forecast_horizon_accuracy(vals, vals, horizon=3)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)

    def test_empty_inputs_zero_evaluated(self) -> None:
        result = forecast_horizon_accuracy([], [], horizon=24)
        assert result["n_evaluated"] == 0

    def test_horizon_caps_evaluation(self) -> None:
        actual = list(range(10))
        predicted = list(range(10))
        result = forecast_horizon_accuracy(actual, predicted, horizon=5)
        assert result["n_evaluated"] == 5


class TestWeightedMovingAverageExtended:
    def test_flat_series_returns_same_value(self) -> None:
        result = weighted_moving_average([5.0] * 5, steps=1)
        assert result[0] == pytest.approx(5.0)

    def test_steps_matches_output_length(self) -> None:
        result = weighted_moving_average([1.0, 2.0, 3.0], steps=3)
        assert len(result) == 3

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            weighted_moving_average([])

    def test_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError):
            weighted_moving_average([1.0, 2.0], weights=[0.0, 0.0])


class TestForecastConfidenceBandExtended:
    def test_empty_residuals_returns_equal_bands(self) -> None:
        forecast = [5.0, 6.0, 7.0]
        result = forecast_confidence_band(forecast, [])
        assert result["lower"] == forecast
        assert result["upper"] == forecast

    def test_bands_bracket_forecast(self) -> None:
        forecast = [10.0, 10.0]
        residuals = [1.0, -1.0, 2.0, -2.0]
        result = forecast_confidence_band(forecast, residuals)
        for lo, up, f in zip(result["lower"], result["upper"], forecast, strict=True):
            assert lo <= f
            assert up >= f

    def test_output_length_matches_forecast(self) -> None:
        forecast = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = forecast_confidence_band(forecast, [0.1, -0.1])
        assert len(result["lower"]) == 5
        assert len(result["upper"]) == 5
