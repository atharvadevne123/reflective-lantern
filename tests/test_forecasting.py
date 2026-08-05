"""Tests for app/forecasting.py."""

from __future__ import annotations

import pytest

from app.forecasting import (
    drift_forecast,
    ensemble_forecast,
    exponential_smoothing_forecast,
    forecast_bias,
    forecast_summary,
    naive_forecast,
    seasonal_naive_forecast,
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


@pytest.mark.parametrize("steps", [1, 3, 6, 12, 24])
def test_naive_forecast_various_steps_value_check(steps: int) -> None:
    from app.forecasting import naive_forecast

    result = naive_forecast(10.0, steps)
    assert len(result) == steps
    assert all(v == pytest.approx(10.0) for v in result)


def test_drift_forecast_constant_returns_last() -> None:
    from app.forecasting import drift_forecast

    values = [5.0, 5.0, 5.0, 5.0]
    result = drift_forecast(values, steps=3)
    assert all(v == pytest.approx(5.0) for v in result)


def test_seasonal_naive_uses_period() -> None:
    from app.forecasting import seasonal_naive_forecast

    pattern = [float(i) for i in range(24)]
    result = seasonal_naive_forecast(pattern * 3, steps=24, period=24)
    assert len(result) == 24


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_exponential_smoothing_various_alpha(alpha: float) -> None:
    from app.forecasting import exponential_smoothing_forecast

    values = [10.0] * 20
    result = exponential_smoothing_forecast(values, steps=5, alpha=alpha)
    assert len(result) == 5
    assert all(v == pytest.approx(10.0, rel=0.01) for v in result)


def test_naive_forecast_zero_steps_empty() -> None:
    from app.forecasting import naive_forecast

    assert naive_forecast(5.0, 0) == []


def test_forecast_bias_returns_float() -> None:
    from app.forecasting import forecast_bias

    result = forecast_bias([10.0, 20.0], [12.0, 18.0])
    assert isinstance(result, float)


def test_forecast_bias_negative_when_under_predicting() -> None:
    from app.forecasting import forecast_bias

    result = forecast_bias([10.0, 10.0], [8.0, 8.0])
    assert result < 0


def test_ensemble_forecast_returns_correct_length() -> None:
    from app.forecasting import ensemble_forecast

    values = [float(i) for i in range(72)]
    result = ensemble_forecast(values, steps=12)
    assert len(result) == 12


def test_ensemble_forecast_constant_input() -> None:
    from app.forecasting import ensemble_forecast

    values = [5.0] * 48
    result = ensemble_forecast(values, steps=6)
    assert len(result) == 6


@pytest.mark.parametrize("steps", [1, 6, 12])
def test_ensemble_forecast_various_steps(steps: int) -> None:
    from app.forecasting import ensemble_forecast

    values = [float(i % 24) for i in range(72)]
    result = ensemble_forecast(values, steps=steps)
    assert len(result) == steps


class TestForecastWithUncertainty:
    def test_returns_dict_with_keys(self) -> None:
        from app.forecasting import forecast_with_uncertainty

        result = forecast_with_uncertainty([1.0, 2.0, 3.0, 4.0, 5.0], horizon=3)
        assert set(result.keys()) == {"point", "lower_80", "upper_80"}

    def test_output_length(self) -> None:
        from app.forecasting import forecast_with_uncertainty

        result = forecast_with_uncertainty([1.0] * 10, horizon=5)
        assert len(result["point"]) == 5
        assert len(result["lower_80"]) == 5
        assert len(result["upper_80"]) == 5

    def test_too_short_raises(self) -> None:
        from app.forecasting import forecast_with_uncertainty

        with pytest.raises(ValueError, match="2 elements"):
            forecast_with_uncertainty([1.0], horizon=3)

    def test_horizon_zero_raises(self) -> None:
        from app.forecasting import forecast_with_uncertainty

        with pytest.raises(ValueError, match="at least 1"):
            forecast_with_uncertainty([1.0, 2.0, 3.0], horizon=0)

    def test_lower_le_upper(self) -> None:
        from app.forecasting import forecast_with_uncertainty

        result = forecast_with_uncertainty(list(range(1, 25)), horizon=6, n_boot=50)
        for lo, hi in zip(result["lower_80"], result["upper_80"]):
            assert lo <= hi


class TestForecastErrorMetrics:
    def test_perfect_forecast(self) -> None:
        from app.forecasting import forecast_error_metrics

        vals = [1.0, 2.0, 3.0, 4.0]
        result = forecast_error_metrics(vals, vals)
        assert result["mae"] == pytest.approx(0.0, abs=1e-6)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_keys_present(self) -> None:
        from app.forecasting import forecast_error_metrics

        result = forecast_error_metrics([1.0, 2.0], [1.1, 2.1])
        assert set(result.keys()) >= {"mae", "rmse", "mape", "bias"}

    def test_empty_raises(self) -> None:
        from app.forecasting import forecast_error_metrics

        with pytest.raises(ValueError, match="empty"):
            forecast_error_metrics([], [])

    def test_length_mismatch_raises(self) -> None:
        from app.forecasting import forecast_error_metrics

        with pytest.raises(ValueError, match="Length mismatch"):
            forecast_error_metrics([1.0, 2.0], [1.0])

    def test_bias_positive_when_over_predicted(self) -> None:
        from app.forecasting import forecast_error_metrics

        result = forecast_error_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert result["bias"] == pytest.approx(1.0, rel=1e-4)


class TestForecastCoverage:
    def test_full_coverage(self) -> None:
        from app.forecasting import forecast_coverage

        actual = [5.0, 5.0, 5.0]
        result = forecast_coverage(actual, [4.0, 4.0, 4.0], [6.0, 6.0, 6.0])
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_no_coverage(self) -> None:
        from app.forecasting import forecast_coverage

        actual = [5.0, 5.0, 5.0]
        result = forecast_coverage(actual, [6.0, 6.0, 6.0], [7.0, 7.0, 7.0])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_empty_raises(self) -> None:
        from app.forecasting import forecast_coverage

        with pytest.raises(ValueError, match="non-empty"):
            forecast_coverage([], [], [])

    def test_length_mismatch_raises(self) -> None:
        from app.forecasting import forecast_coverage

        with pytest.raises(ValueError, match="same length"):
            forecast_coverage([1.0, 2.0], [0.5, 1.5, 2.5], [1.5, 2.5, 3.5])

    def test_result_in_range(self) -> None:
        from app.forecasting import forecast_coverage

        actual = [float(i) for i in range(10)]
        lower = [float(i) - 2 for i in range(10)]
        upper = [float(i) + 2 for i in range(10)]
        result = forecast_coverage(actual, lower, upper)
        assert 0.0 <= result <= 1.0
