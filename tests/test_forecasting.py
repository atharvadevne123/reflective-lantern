"""Tests for app/forecasting.py."""

from __future__ import annotations

import pytest

from app.forecasting import (
    confidence_interval,
    drift_forecast,
    ensemble_forecast,
    exponential_smoothing_forecast,
    forecast_bias,
    forecast_summary,
    naive_forecast,
    seasonal_naive_forecast,
    stepwise_error_growth,
    weighted_ensemble_forecast,
)

HISTORY = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0]


def test_naive_forecast_length() -> None:
    result = naive_forecast(10.0, steps=5)
    assert len(result) == 5


def test_naive_forecast_constant() -> None:
    result = naive_forecast(7.5, steps=3)
    assert all(v == pytest.approx(7.5) for v in result)


def test_naive_forecast_zero_steps() -> None:
    assert naive_forecast(10.0, steps=0) == []


def test_naive_forecast_negative_steps() -> None:
    assert naive_forecast(10.0, steps=-1) == []


def test_drift_forecast_length() -> None:
    result = drift_forecast(HISTORY, steps=4)
    assert len(result) == 4


def test_drift_forecast_rising_history() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = drift_forecast(values, steps=2)
    assert result[1] > result[0]


def test_drift_forecast_single_value() -> None:
    result = drift_forecast([10.0], steps=3)
    assert len(result) == 3
    assert all(v == pytest.approx(10.0) for v in result)


def test_drift_forecast_empty() -> None:
    assert drift_forecast([], steps=3) == []


def test_seasonal_naive_forecast_length() -> None:
    result = seasonal_naive_forecast(HISTORY, steps=3, period=3)
    assert len(result) == 3


def test_seasonal_naive_forecast_repeats_pattern() -> None:
    values = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    result = seasonal_naive_forecast(values, steps=3, period=3)
    assert len(result) == 3


def test_seasonal_naive_forecast_empty() -> None:
    assert seasonal_naive_forecast([], steps=3) == []


def test_exponential_smoothing_forecast_length() -> None:
    result = exponential_smoothing_forecast(HISTORY, steps=5)
    assert len(result) == 5


def test_exponential_smoothing_forecast_all_equal() -> None:
    result = exponential_smoothing_forecast(HISTORY, steps=3, alpha=0.3)
    assert result[0] == result[1] == result[2]


def test_exponential_smoothing_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        exponential_smoothing_forecast(HISTORY, steps=3, alpha=0.0)


def test_exponential_smoothing_alpha_1() -> None:
    result = exponential_smoothing_forecast([10.0, 20.0, 30.0], steps=2, alpha=1.0)
    assert all(v == pytest.approx(30.0) for v in result)


def test_forecast_summary_correct() -> None:
    forecasts = [10.0, 20.0, 30.0]
    s = forecast_summary(forecasts)
    assert s["mean"] == pytest.approx(20.0)
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    assert s["total"] == pytest.approx(60.0)
    assert s["steps"] == 3


def test_forecast_summary_empty() -> None:
    s = forecast_summary([])
    assert s["steps"] == 0
    assert s["mean"] == 0.0


@pytest.mark.parametrize("steps", [1, 3, 6, 24])
def test_naive_forecast_various_steps(steps) -> None:
    result = naive_forecast(15.0, steps=steps)
    assert len(result) == steps


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_exponential_smoothing_valid_alphas(alpha: float) -> None:
    result = exponential_smoothing_forecast([5.0, 10.0, 15.0], steps=3, alpha=alpha)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


@pytest.mark.parametrize("steps", [1, 2, 5, 10])
def test_drift_forecast_various_steps(steps: int) -> None:
    values = [float(i) for i in range(1, 8)]
    result = drift_forecast(values, steps=steps)
    assert len(result) == steps


@pytest.mark.parametrize("period", [4, 7, 12, 24])
def test_seasonal_naive_various_periods(period: int) -> None:
    values = [float(i % period) for i in range(period * 3)]
    result = seasonal_naive_forecast(values, steps=period, period=period)
    assert len(result) == period


def test_naive_forecast_value_preserved() -> None:
    result = naive_forecast(42.0, steps=1)
    assert result[0] == pytest.approx(42.0)


def test_drift_forecast_zero_steps() -> None:
    assert drift_forecast([1.0, 2.0, 3.0], steps=0) == []


def test_forecast_summary_single_value() -> None:
    s = forecast_summary([7.5])
    assert s["mean"] == pytest.approx(7.5)
    assert s["min"] == 7.5
    assert s["max"] == 7.5
    assert s["steps"] == 1


@pytest.mark.parametrize("last,steps", [(0.0, 1), (100.0, 5), (0.001, 10)])
def test_naive_forecast_boundary_values(last: float, steps: int) -> None:
    result = naive_forecast(last, steps=steps)
    assert len(result) == steps
    assert all(v == pytest.approx(last) for v in result)


def test_drift_forecast_declining_history() -> None:
    values = [10.0, 8.0, 6.0, 4.0, 2.0]
    result = drift_forecast(values, steps=3)
    assert result[0] < values[-1]
    assert result[1] < result[0]


def test_drift_forecast_flat_history() -> None:
    values = [5.0, 5.0, 5.0, 5.0]
    result = drift_forecast(values, steps=4)
    assert all(v == pytest.approx(5.0) for v in result)


@pytest.mark.parametrize("alpha", [0.01, 1.0])
def test_exponential_smoothing_boundary_alphas(alpha: float) -> None:
    result = exponential_smoothing_forecast([1.0, 2.0, 3.0], steps=2, alpha=alpha)
    assert len(result) == 2
    assert all(v > 0 for v in result)


def test_forecast_summary_total_matches_sum() -> None:
    forecasts = [2.5, 3.5, 4.0]
    s = forecast_summary(forecasts)
    assert s["total"] == pytest.approx(sum(forecasts), rel=1e-4)


def test_seasonal_naive_forecast_period_larger_than_history() -> None:
    values = [1.0, 2.0, 3.0]
    result = seasonal_naive_forecast(values, steps=5, period=24)
    assert len(result) == 5


def test_ensemble_forecast_length() -> None:
    result = ensemble_forecast(HISTORY, steps=5)
    assert len(result) == 5


def test_ensemble_forecast_empty_values() -> None:
    assert ensemble_forecast([], steps=5) == []


def test_ensemble_forecast_zero_steps() -> None:
    assert ensemble_forecast(HISTORY, steps=0) == []


def test_ensemble_forecast_values_in_range() -> None:
    result = ensemble_forecast([10.0] * 30, steps=5)
    assert all(8.0 <= v <= 12.0 for v in result)


def test_forecast_bias_positive() -> None:
    actual = [10.0, 10.0, 10.0]
    predicted = [12.0, 12.0, 12.0]
    assert forecast_bias(actual, predicted) == pytest.approx(2.0)


def test_forecast_bias_negative() -> None:
    actual = [10.0, 10.0]
    predicted = [8.0, 8.0]
    assert forecast_bias(actual, predicted) == pytest.approx(-2.0)


def test_forecast_bias_zero() -> None:
    actual = [5.0, 7.0]
    predicted = [7.0, 5.0]
    assert forecast_bias(actual, predicted) == pytest.approx(0.0)


def test_forecast_bias_empty_raises() -> None:
    with pytest.raises(ValueError):
        forecast_bias([], [1.0])


@pytest.mark.parametrize("w", [(0.5, 0.3, 0.2), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_ensemble_forecast_custom_weights(w) -> None:
    result = ensemble_forecast(HISTORY, steps=3, weights=w)
    assert len(result) == 3


def test_confidence_interval_length() -> None:
    fc = [10.0, 11.0, 12.0]
    result = confidence_interval(fc, std_error=1.0)
    assert len(result) == 3


def test_confidence_interval_keys() -> None:
    result = confidence_interval([10.0], std_error=2.0)
    assert "point" in result[0]
    assert "lower" in result[0]
    assert "upper" in result[0]


def test_confidence_interval_bounds() -> None:
    fc = [10.0]
    result = confidence_interval(fc, std_error=2.0, z_score=1.96)
    assert result[0]["lower"] < result[0]["point"] < result[0]["upper"]


def test_confidence_interval_zero_std() -> None:
    fc = [5.0, 6.0]
    result = confidence_interval(fc, std_error=0.0)
    assert all(r["lower"] == r["point"] == r["upper"] for r in result)


@pytest.mark.parametrize("z_score", [1.0, 1.645, 1.96, 2.576])
def test_confidence_interval_z_parametrize(z_score) -> None:
    fc = [10.0]
    result = confidence_interval(fc, std_error=1.0, z_score=z_score)
    assert result[0]["upper"] - result[0]["lower"] == pytest.approx(2 * z_score, rel=1e-4)


def test_stepwise_error_growth_length() -> None:
    result = stepwise_error_growth([1.0, 2.0, 3.0], base_error=1.0)
    assert len(result) == 3


def test_stepwise_error_growth_first_step() -> None:
    result = stepwise_error_growth([1.0], base_error=2.0, growth_factor=1.1)
    assert result[0] == pytest.approx(2.0)


def test_stepwise_error_growth_increasing() -> None:
    result = stepwise_error_growth([1.0] * 5, base_error=1.0, growth_factor=1.1)
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_weighted_ensemble_forecast_basic() -> None:
    f1 = [1.0, 2.0, 3.0]
    f2 = [3.0, 4.0, 5.0]
    result = weighted_ensemble_forecast([f1, f2], weights=[1.0, 1.0])
    assert result == pytest.approx([2.0, 3.0, 4.0])


def test_weighted_ensemble_forecast_unequal_weights() -> None:
    f1 = [10.0]
    f2 = [0.0]
    result = weighted_ensemble_forecast([f1, f2], weights=[1.0, 0.0])
    assert result == pytest.approx([10.0])


def test_weighted_ensemble_forecast_empty_raises() -> None:
    with pytest.raises(ValueError):
        weighted_ensemble_forecast([], [])


def test_weighted_ensemble_forecast_zero_weights_raises() -> None:
    with pytest.raises(ValueError):
        weighted_ensemble_forecast([[1.0]], [0.0])


def test_weighted_ensemble_forecast_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        weighted_ensemble_forecast([[1.0, 2.0], [3.0]], [1.0, 1.0])
