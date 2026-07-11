"""Tests for time-series forecasting module."""

import math

import pytest

from app.time_series import (
    compute_sma,
    exponential_smoothing_forecast,
    linear_trend_forecast,
)


def test_sma_length_matches_input() -> None:
    result = compute_sma([1, 2, 3, 4, 5], window=3)
    assert len(result) == 5


def test_sma_first_values_nan() -> None:
    result = compute_sma([1, 2, 3, 4, 5], window=3)
    assert math.isnan(result[0])
    assert math.isnan(result[1])


def test_sma_correct_value() -> None:
    result = compute_sma([100_000, 200_000, 300_000, 400_000], window=3)
    assert result[2] == pytest.approx(200_000.0)
    assert result[3] == pytest.approx(300_000.0)


def test_sma_too_short_all_nan() -> None:
    result = compute_sma([100, 200], window=5)
    assert all(math.isnan(v) for v in result)


def test_linear_trend_positive_slope() -> None:
    values = [100_000, 110_000, 120_000, 130_000, 140_000]
    result = linear_trend_forecast(values, horizon=2)
    assert result["slope"] > 0
    assert len(result["forecasts"]) == 2
    assert result["r_squared"] > 0.99


def test_linear_trend_flat_r_squared() -> None:
    values = [500_000] * 5
    result = linear_trend_forecast(values, horizon=3)
    assert result["slope"] == pytest.approx(0.0, abs=1.0)
    assert result["r_squared"] == 0.0


def test_linear_trend_too_few_values() -> None:
    result = linear_trend_forecast([100_000, 120_000], horizon=3)
    assert result["forecasts"] == []
    assert result["r_squared"] == 0.0


def test_linear_trend_forecast_extrapolates() -> None:
    values = [100_000 + i * 10_000 for i in range(5)]
    result = linear_trend_forecast(values, horizon=3)
    for f in result["forecasts"]:
        assert f > values[-1]


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.7, 0.9])
def test_exp_smoothing_returns_horizon_values(alpha) -> None:
    values = [100_000, 110_000, 120_000, 130_000]
    result = exponential_smoothing_forecast(values, alpha=alpha, horizon=3)
    assert len(result) == 3


def test_exp_smoothing_empty_input() -> None:
    assert exponential_smoothing_forecast([], horizon=3) == []


def test_exp_smoothing_single_value() -> None:
    result = exponential_smoothing_forecast([500_000], horizon=2)
    assert all(v == pytest.approx(500_000.0) for v in result)


def test_sma_window_one_returns_input() -> None:
    values = [100.0, 200.0, 300.0]
    result = compute_sma(values, window=1)
    assert result == pytest.approx(values)


@pytest.mark.parametrize("horizon", [1, 5, 12])
def test_linear_trend_forecast_horizon_length(horizon) -> None:
    values = [100_000 + i * 5_000 for i in range(8)]
    result = linear_trend_forecast(values, horizon=horizon)
    assert len(result["forecasts"]) == horizon


def test_linear_trend_negative_slope() -> None:
    values = [200_000, 180_000, 160_000, 140_000, 120_000]
    result = linear_trend_forecast(values, horizon=2)
    assert result["slope"] < 0


def test_exp_smoothing_high_alpha_tracks_recent() -> None:
    # With alpha near 1.0, latest value dominates
    values = [100_000] * 9 + [500_000]
    result = exponential_smoothing_forecast(values, alpha=0.99, horizon=1)
    assert result[0] > 400_000


@pytest.mark.parametrize("n", [3, 6, 10])
def test_sma_valid_values_count(n) -> None:
    values = list(range(n))
    result = compute_sma(values, window=3)
    valid_count = sum(1 for v in result if not math.isnan(v))
    assert valid_count == max(0, n - 2)


def test_linear_trend_returns_required_keys() -> None:
    result = linear_trend_forecast([100_000, 110_000, 120_000], horizon=2)
    for key in ("slope", "intercept", "r_squared", "forecasts"):
        assert key in result


@pytest.mark.parametrize(
    "alpha,expected_smoothed_close_to",
    [(0.01, 100_000), (0.99, 200_000)],
)
def test_exp_smoothing_alpha_effect(alpha, expected_smoothed_close_to) -> None:
    values = [100_000] * 8 + [200_000]
    result = exponential_smoothing_forecast(values, alpha=alpha, horizon=1)
    diff = abs(result[0] - expected_smoothed_close_to)
    assert diff < 150_000


def test_sma_all_same_values() -> None:
    values = [250_000.0] * 6
    result = compute_sma(values, window=3)
    for v in result[2:]:
        assert v == pytest.approx(250_000.0)


def test_linear_trend_r_squared_bounded() -> None:
    values = [100_000 + i * 10_000 for i in range(10)]
    result = linear_trend_forecast(values)
    assert 0.0 <= result["r_squared"] <= 1.0


def test_sma_window_zero_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        compute_sma([1.0, 2.0, 3.0], window=0)


def test_exp_smoothing_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        exponential_smoothing_forecast([1.0, 2.0], alpha=0.0)


def test_exp_smoothing_alpha_above_one_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        exponential_smoothing_forecast([1.0, 2.0], alpha=1.5)


@pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0])
def test_exp_smoothing_valid_alpha_boundary(alpha: float) -> None:
    result = exponential_smoothing_forecast([100_000.0, 200_000.0], alpha=alpha, horizon=2)
    assert len(result) == 2


def test_sma_window_equals_series_length() -> None:
    values = [100.0, 200.0, 300.0]
    result = compute_sma(values, window=3)
    assert not any(v != v for v in result[2:])  # no NaN at last position
    assert result[2] == pytest.approx(200.0)


def test_linear_trend_intercept_positive_for_ascending() -> None:
    values = [100_000 + i * 10_000 for i in range(5)]
    result = linear_trend_forecast(values, horizon=1)
    assert result["intercept"] > 0


def test_min_periods_constant() -> None:
    from app.time_series import MIN_PERIODS

    assert MIN_PERIODS >= 2


def test_max_horizon_constant() -> None:
    from app.time_series import MAX_HORIZON

    assert MAX_HORIZON > 0


def test_compute_sma_raises_on_zero_window() -> None:
    from app.time_series import compute_sma

    with pytest.raises(ValueError):
        compute_sma([100.0, 200.0, 300.0], window=0)


def test_exp_smoothing_raises_on_zero_alpha() -> None:
    from app.time_series import exponential_smoothing_forecast

    with pytest.raises(ValueError):
        exponential_smoothing_forecast([100.0, 200.0], alpha=0.0, horizon=1)


def test_exp_smoothing_raises_on_alpha_above_one() -> None:
    from app.time_series import exponential_smoothing_forecast

    with pytest.raises(ValueError):
        exponential_smoothing_forecast([100.0, 200.0], alpha=1.1, horizon=1)


@pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0])
def test_exp_smoothing_valid_alpha_range(alpha: float) -> None:
    from app.time_series import exponential_smoothing_forecast

    result = exponential_smoothing_forecast([100_000.0, 120_000.0], alpha=alpha, horizon=3)
    assert len(result) == 3
