"""Tests for app/forecasting.py."""

from __future__ import annotations

import pytest

from app.forecasting import (
    confidence_interval,
    drift_forecast,
    ensemble_forecast,
    exponential_smoothing_forecast,
    forecast_bias,
    forecast_error_metrics,
    forecast_residuals,
    forecast_summary,
    horizon_degradation,
    mae_score,
    mape_score,
    naive_forecast,
    rmse_score,
    seasonal_naive_forecast,
    stepwise_error_growth,
    weighted_ensemble_forecast,
    weighted_forecast,
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
        for lo, hi in zip(result["lower_80"], result["upper_80"], strict=False):
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


def test_mae_score_perfect() -> None:
    assert mae_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_mae_score_constant_error() -> None:
    assert mae_score([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_mae_score_empty_raises() -> None:
    with pytest.raises(ValueError):
        mae_score([], [])


def test_mae_score_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mae_score([1.0], [1.0, 2.0])


def test_mae_score_positive() -> None:
    assert mae_score([10.0, 20.0], [12.0, 18.0]) > 0


def test_rmse_score_perfect() -> None:
    assert rmse_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_rmse_score_constant_error() -> None:
    assert rmse_score([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_rmse_score_empty_raises() -> None:
    with pytest.raises(ValueError):
        rmse_score([], [])


def test_rmse_score_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rmse_score([1.0], [1.0, 2.0])


def test_rmse_score_gte_mae() -> None:
    actual = [10.0, 20.0, 30.0, 5.0]
    predicted = [12.0, 18.0, 35.0, 5.0]
    assert rmse_score(actual, predicted) >= mae_score(actual, predicted)


@pytest.mark.parametrize("err", [1.0, 2.0, 5.0])
def test_mae_score_constant_parametrize(err: float) -> None:
    n = 5
    actual = [0.0] * n
    predicted = [err] * n
    assert mae_score(actual, predicted) == pytest.approx(err)


class TestForecastIntervalHitRate:
    def test_perfect_coverage(self) -> None:
        from app.forecasting import forecast_interval_hit_rate

        actual = [1.0, 2.0, 3.0]
        lower = [0.5, 1.5, 2.5]
        upper = [1.5, 2.5, 3.5]
        assert forecast_interval_hit_rate(actual, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        from app.forecasting import forecast_interval_hit_rate

        actual = [10.0, 20.0, 30.0]
        lower = [0.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0]
        assert forecast_interval_hit_rate(actual, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        from app.forecasting import forecast_interval_hit_rate

        actual = [1.0, 10.0, 2.0, 10.0]
        lower = [0.0, 0.0, 0.0, 0.0]
        upper = [5.0, 5.0, 5.0, 5.0]
        assert forecast_interval_hit_rate(actual, lower, upper) == pytest.approx(0.5)

    def test_empty_raises(self) -> None:
        from app.forecasting import forecast_interval_hit_rate

        with pytest.raises(ValueError):
            forecast_interval_hit_rate([], [], [])

    def test_length_mismatch_raises(self) -> None:
        from app.forecasting import forecast_interval_hit_rate

        with pytest.raises(ValueError):
            forecast_interval_hit_rate([1.0, 2.0], [0.0], [5.0])


class TestQuantileForecast:
    def test_default_quantiles(self) -> None:
        from app.forecasting import quantile_forecast

        result = quantile_forecast([1.0, 2.0, 3.0, 4.0, 5.0], steps=3)
        assert set(result.keys()) == {"q10", "q50", "q90"}

    def test_step_count(self) -> None:
        from app.forecasting import quantile_forecast

        result = quantile_forecast([1.0, 2.0, 3.0, 4.0, 5.0], steps=5)
        for values in result.values():
            assert len(values) == 5

    def test_too_few_history_raises(self) -> None:
        from app.forecasting import quantile_forecast

        with pytest.raises(ValueError, match="at least 2"):
            quantile_forecast([1.0], steps=3)

    def test_invalid_quantile_raises(self) -> None:
        from app.forecasting import quantile_forecast

        with pytest.raises(ValueError, match="quantiles"):
            quantile_forecast([1.0, 2.0, 3.0], steps=2, quantiles=[0.0, 0.5])

    def test_zero_steps_raises(self) -> None:
        from app.forecasting import quantile_forecast

        with pytest.raises(ValueError, match="steps"):
            quantile_forecast([1.0, 2.0, 3.0], steps=0)

    @pytest.mark.parametrize("steps", [1, 5, 10])
    def test_various_step_counts(self, steps: int) -> None:
        from app.forecasting import quantile_forecast

        result = quantile_forecast(list(range(1, 11)), steps=steps)
        for vals in result.values():
            assert len(vals) == steps


class TestForecastDirection:
    def test_up(self) -> None:
        from app.forecasting import forecast_direction

        assert forecast_direction([110.0, 120.0, 130.0], baseline=100.0) == "up"

    def test_down(self) -> None:
        from app.forecasting import forecast_direction

        assert forecast_direction([90.0, 85.0, 80.0], baseline=100.0) == "down"

    def test_flat(self) -> None:
        from app.forecasting import forecast_direction

        assert forecast_direction([100.5, 99.5, 100.0], baseline=100.0) == "flat"

    def test_empty_raises(self) -> None:
        from app.forecasting import forecast_direction

        with pytest.raises(ValueError):
            forecast_direction([], baseline=100.0)

    def test_zero_baseline(self) -> None:
        from app.forecasting import forecast_direction

        assert forecast_direction([0.0, 0.0], baseline=0.0) == "flat"

    @pytest.mark.parametrize(
        "forecast,baseline,expected",
        [
            ([200.0], 100.0, "up"),
            ([50.0], 100.0, "down"),
            ([100.0], 100.0, "flat"),
        ],
    )
    def test_parametrized(self, forecast: list, baseline: float, expected: str) -> None:
        from app.forecasting import forecast_direction

        assert forecast_direction(forecast, baseline) == expected


class TestConsecutiveMissCount:
    def test_no_misses(self) -> None:
        from app.forecasting import consecutive_miss_count

        assert consecutive_miss_count([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], threshold=0.1) == 0

    def test_all_misses(self) -> None:
        from app.forecasting import consecutive_miss_count

        assert consecutive_miss_count([1.0, 2.0, 3.0], [5.0, 6.0, 7.0], threshold=0.1) == 3

    def test_run_detection(self) -> None:
        from app.forecasting import consecutive_miss_count

        actual = [1.0, 1.0, 1.0, 1.0, 1.0]
        predicted = [5.0, 5.0, 1.0, 5.0, 1.0]
        assert consecutive_miss_count(actual, predicted, threshold=0.5) == 2

    def test_empty_raises(self) -> None:
        from app.forecasting import consecutive_miss_count

        with pytest.raises(ValueError):
            consecutive_miss_count([], [], threshold=1.0)

    def test_length_mismatch_raises(self) -> None:
        from app.forecasting import consecutive_miss_count

        with pytest.raises(ValueError):
            consecutive_miss_count([1.0, 2.0], [1.0], threshold=1.0)

    def test_negative_threshold_raises(self) -> None:
        from app.forecasting import consecutive_miss_count

        with pytest.raises(ValueError, match="non-negative"):
            consecutive_miss_count([1.0], [1.0], threshold=-1.0)


# ---------------------------------------------------------------------------
# Tests for forecast_confidence_interval, weighted_ensemble_forecast, horizon_degradation
# ---------------------------------------------------------------------------


class TestForecastConfidenceInterval:
    def test_basic_symmetry(self) -> None:
        from app.forecasting import forecast_confidence_interval

        result = forecast_confidence_interval([100.0], residual_std=10.0, z=1.0)
        assert result[0][0] == pytest.approx(90.0, abs=0.01)
        assert result[0][1] == pytest.approx(110.0, abs=0.01)

    def test_zero_std_is_point_forecast(self) -> None:
        from app.forecasting import forecast_confidence_interval

        result = forecast_confidence_interval([50.0, 60.0], residual_std=0.0)
        assert all(lo == hi for lo, hi in result)

    def test_negative_std_raises(self) -> None:
        import pytest

        from app.forecasting import forecast_confidence_interval

        with pytest.raises(ValueError):
            forecast_confidence_interval([1.0], residual_std=-1.0)

    def test_invalid_z_raises(self) -> None:
        import pytest

        from app.forecasting import forecast_confidence_interval

        with pytest.raises(ValueError):
            forecast_confidence_interval([1.0], residual_std=1.0, z=0.0)


class TestWeightedEnsembleForecastNew:
    def test_equal_weights(self) -> None:
        from app.forecasting import weighted_ensemble_forecast

        f1 = [10.0, 20.0]
        f2 = [20.0, 40.0]
        result = weighted_ensemble_forecast([f1, f2])
        assert result == [pytest.approx(15.0), pytest.approx(30.0)]

    def test_custom_weights(self) -> None:
        from app.forecasting import weighted_ensemble_forecast

        result = weighted_ensemble_forecast([[100.0], [200.0]], weights=[0.8, 0.2])
        assert result[0] == pytest.approx(120.0, abs=0.01)

    def test_empty_forecasts_raises(self) -> None:
        import pytest

        from app.forecasting import weighted_ensemble_forecast

        with pytest.raises(ValueError):
            weighted_ensemble_forecast([])

    def test_negative_weights_raises(self) -> None:
        import pytest

        from app.forecasting import weighted_ensemble_forecast

        with pytest.raises(ValueError):
            weighted_ensemble_forecast([[1.0, 2.0], [3.0, 4.0]], weights=[-0.5, 0.5])


class TestHorizonDegradation:
    def test_positive_slope_for_growing_errors(self) -> None:
        from app.forecasting import horizon_degradation

        errors = [1.0, 2.0, 3.0, 4.0]
        assert horizon_degradation(errors) > 0.0

    def test_flat_errors_zero_slope(self) -> None:
        from app.forecasting import horizon_degradation

        assert horizon_degradation([2.0, 2.0, 2.0, 2.0]) == pytest.approx(0.0, abs=1e-6)

    def test_too_short_raises(self) -> None:
        import pytest

        from app.forecasting import horizon_degradation

        with pytest.raises(ValueError):
            horizon_degradation([1.0])


@pytest.mark.parametrize("steps", [1, 3, 5, 10])
def test_naive_forecast_step_count(steps: int) -> None:
    result = naive_forecast(7.5, steps=steps)
    assert len(result) == steps
    assert all(v == 7.5 for v in result)


@pytest.mark.parametrize(
    "actual,predicted,expected_bias_sign",
    [
        ([10.0, 10.0, 10.0], [11.0, 11.0, 11.0], 1),
        ([10.0, 10.0, 10.0], [9.0, 9.0, 9.0], -1),
        ([10.0, 10.0], [10.0, 10.0], 0),
    ],
)
def test_forecast_bias_sign(actual: list[float], predicted: list[float], expected_bias_sign: int) -> None:
    bias = forecast_bias(actual, predicted)
    if expected_bias_sign > 0:
        assert bias > 0
    elif expected_bias_sign < 0:
        assert bias < 0
    else:
        assert bias == pytest.approx(0.0, abs=1e-9)


def test_forecast_residuals_zero_for_perfect_prediction() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    residuals = forecast_residuals(actual, actual)
    assert all(r == pytest.approx(0.0) for r in residuals)


def test_forecast_error_metrics_perfect_prediction() -> None:
    actual = [5.0, 10.0, 15.0]
    metrics = forecast_error_metrics(actual, actual)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
def test_exponential_smoothing_length(alpha: float) -> None:
    result = exponential_smoothing_forecast(HISTORY, steps=4, alpha=alpha)
    assert len(result) == 4


def test_mape_score_perfect_prediction() -> None:
    actual = [10.0, 20.0, 30.0]
    assert mape_score(actual, actual) == pytest.approx(0.0, abs=1e-6)


def test_weighted_forecast_equal_weights() -> None:
    result = weighted_forecast([10.0, 20.0, 30.0], [1 / 3, 1 / 3, 1 / 3])
    assert result == pytest.approx(20.0, abs=1e-4)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_mae_score_zero_for_perfect_prediction(n: int) -> None:
    actual = [float(i) for i in range(n)]
    assert mae_score(actual, actual) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("n", [5, 10, 20])
def test_rmse_score_zero_for_perfect_prediction(n: int) -> None:
    actual = [float(i) for i in range(n)]
    assert rmse_score(actual, actual) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        ([10.0, 10.0, 10.0], [12.0, 12.0, 12.0]),
    ],
)
def test_forecast_bias_positive_when_over_predicted(actual: list, predicted: list) -> None:
    from app.forecasting import bias_score

    result = bias_score(actual, predicted)
    assert result > 0.0


@pytest.mark.parametrize("horizon", [3, 6, 12])
def test_horizon_degradation_length(horizon: int) -> None:
    values = [float(i + 1) for i in range(30)]
    result = horizon_degradation(values, horizon=horizon)
    assert len(result) == horizon


class TestDirectionalAccuracy:
    def test_perfect_directional_accuracy(self) -> None:
        from app.forecasting import directional_accuracy

        result = directional_accuracy([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result == pytest.approx(1.0)

    def test_short_series_returns_zero(self) -> None:
        from app.forecasting import directional_accuracy

        assert directional_accuracy([1.0], [1.0]) == 0.0

    def test_opposite_directions_zero(self) -> None:
        from app.forecasting import directional_accuracy

        result = directional_accuracy([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert result == pytest.approx(0.0)

    def test_raises_on_length_mismatch(self) -> None:
        from app.forecasting import directional_accuracy

        with pytest.raises(ValueError):
            directional_accuracy([1.0, 2.0], [1.0])


class TestForecastBiasRatio:
    def test_unbiased_returns_one(self) -> None:
        from app.forecasting import forecast_bias_ratio

        assert forecast_bias_ratio([2.0, 4.0], [2.0, 4.0]) == pytest.approx(1.0)

    def test_zero_actual_returns_zero(self) -> None:
        from app.forecasting import forecast_bias_ratio

        assert forecast_bias_ratio([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_overprediction_ratio_gt_one(self) -> None:
        from app.forecasting import forecast_bias_ratio

        result = forecast_bias_ratio([1.0, 2.0], [2.0, 4.0])
        assert result > 1.0

    def test_raises_on_length_mismatch(self) -> None:
        from app.forecasting import forecast_bias_ratio

        with pytest.raises(ValueError):
            forecast_bias_ratio([1.0, 2.0], [1.0])


class TestMeanPercentageError:
    def test_over_forecast(self) -> None:
        from app.forecasting import mean_percentage_error
        assert mean_percentage_error([10.0, 20.0], [11.0, 22.0]) == pytest.approx(10.0)

    def test_under_forecast(self) -> None:
        from app.forecasting import mean_percentage_error
        assert mean_percentage_error([10.0, 20.0], [9.0, 18.0]) == pytest.approx(-10.0)

    def test_perfect_forecast(self) -> None:
        from app.forecasting import mean_percentage_error
        assert mean_percentage_error([5.0, 10.0], [5.0, 10.0]) == pytest.approx(0.0)

    def test_zero_in_actual_raises(self) -> None:
        from app.forecasting import mean_percentage_error
        with pytest.raises(ValueError, match="zeros"):
            mean_percentage_error([0.0, 10.0], [1.0, 10.0])

    def test_length_mismatch_raises(self) -> None:
        from app.forecasting import mean_percentage_error
        with pytest.raises(ValueError):
            mean_percentage_error([1.0, 2.0], [1.0])


class TestPeakForecastHour:
    def test_peak_at_end(self) -> None:
        from app.forecasting import peak_forecast_hour
        assert peak_forecast_hour([1.0, 2.0, 3.0]) == 2

    def test_peak_in_middle(self) -> None:
        from app.forecasting import peak_forecast_hour
        assert peak_forecast_hour([1.0, 5.0, 2.0]) == 1

    def test_empty_raises(self) -> None:
        from app.forecasting import peak_forecast_hour
        with pytest.raises(ValueError):
            peak_forecast_hour([])


class TestForecastVolatility:
    def test_constant_series(self) -> None:
        from app.forecasting import forecast_volatility
        assert forecast_volatility([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_positive_volatility(self) -> None:
        from app.forecasting import forecast_volatility
        result = forecast_volatility([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result > 0.0

    def test_empty_raises(self) -> None:
        from app.forecasting import forecast_volatility
        with pytest.raises(ValueError):
            forecast_volatility([])
