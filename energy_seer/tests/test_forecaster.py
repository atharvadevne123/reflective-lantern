"""Tests for energy_seer/app/forecaster.py."""

from __future__ import annotations

import pytest


class TestLinearTrend:
    def test_increasing_series_positive_slope(self) -> None:
        from app.forecaster import linear_trend

        result = linear_trend([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["slope"] > 0

    def test_flat_series_slope_near_zero(self) -> None:
        from app.forecaster import linear_trend

        result = linear_trend([5.0, 5.0, 5.0, 5.0])
        assert abs(result["slope"]) < 0.01

    def test_single_value_returns_zero_slope(self) -> None:
        from app.forecaster import linear_trend

        result = linear_trend([3.0])
        assert result["slope"] == 0.0

    def test_returns_required_keys(self) -> None:
        from app.forecaster import linear_trend

        result = linear_trend([1.0, 2.0, 3.0])
        assert {"slope", "intercept", "next"} <= result.keys()


class TestMovingAverageForecast:
    def test_single_step_returns_one_value(self) -> None:
        from app.forecaster import moving_average_forecast

        result = moving_average_forecast([1.0, 2.0, 3.0], steps=1)
        assert len(result) == 1

    def test_multiple_steps_length(self) -> None:
        from app.forecaster import moving_average_forecast

        result = moving_average_forecast([1.0, 2.0, 3.0], steps=5)
        assert len(result) == 5

    def test_flat_series_forecast_equals_value(self) -> None:
        from app.forecaster import moving_average_forecast

        result = moving_average_forecast([4.0] * 10, steps=3)
        assert all(abs(v - 4.0) < 0.01 for v in result)


class TestSeasonalDecompose:
    def test_insufficient_data_returns_empty(self) -> None:
        from app.forecaster import seasonal_decompose_simple

        result = seasonal_decompose_simple([1.0, 2.0], period=24)
        assert result["seasonal"] == []

    def test_output_lengths_match_input(self) -> None:
        from app.forecaster import seasonal_decompose_simple

        values = list(range(48))
        result = seasonal_decompose_simple(values, period=24)
        assert len(result["trend"]) == len(values)

    def test_seasonal_component_length_equals_period(self) -> None:
        from app.forecaster import seasonal_decompose_simple

        values = list(range(48))
        result = seasonal_decompose_simple(values, period=24)
        assert len(result["seasonal"]) == 24


class TestWeightedMovingAverage:
    def test_flat_series_returns_same_value(self) -> None:
        from app.forecaster import weighted_moving_average

        result = weighted_moving_average([5.0] * 10, steps=1)
        assert abs(result[0] - 5.0) < 0.01

    def test_steps_length(self) -> None:
        from app.forecaster import weighted_moving_average

        result = weighted_moving_average([1.0, 2.0, 3.0], steps=4)
        assert len(result) == 4

    def test_empty_raises(self) -> None:
        from app.forecaster import weighted_moving_average

        import pytest

        with pytest.raises(ValueError):
            weighted_moving_average([])

    def test_custom_weights(self) -> None:
        from app.forecaster import weighted_moving_average

        result = weighted_moving_average([2.0, 4.0], weights=[1, 3], steps=1)
        assert result[0] == pytest.approx(3.5)


class TestForecastHorizonAccuracy:
    def test_perfect_forecast_zero_mae(self) -> None:
        from app.forecaster import forecast_horizon_accuracy

        actual = [1.0, 2.0, 3.0, 4.0]
        result = forecast_horizon_accuracy(actual, actual, horizon=4)
        assert result["mae"] == pytest.approx(0.0)

    def test_empty_inputs_zero_evaluated(self) -> None:
        from app.forecaster import forecast_horizon_accuracy

        result = forecast_horizon_accuracy([], [], horizon=10)
        assert result["n_evaluated"] == 0

    @pytest.mark.parametrize("horizon", [1, 5, 10, 24])
    def test_n_evaluated_bounded_by_horizon(self, horizon: int) -> None:
        from app.forecaster import forecast_horizon_accuracy

        actual = [1.0] * 30
        predicted = [1.0] * 30
        result = forecast_horizon_accuracy(actual, predicted, horizon=horizon)
        assert result["n_evaluated"] <= horizon
