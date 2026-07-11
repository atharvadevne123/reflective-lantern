"""Time-series forecasting utility tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.time_series import (
    detect_spikes,
    forecast_linear_trend,
    seasonal_baseline,
    simple_moving_average,
)


def test_sma_length():
    result = simple_moving_average([1.0] * 10, window=3)
    assert len(result) == 10


def test_sma_flat_series():
    result = simple_moving_average([5.0] * 20, window=5)
    assert all(abs(v - 5.0) < 0.1 for v in result[-10:])


def test_seasonal_baseline_length():
    data = list(range(48))
    baseline = seasonal_baseline(data, period=24)
    assert len(baseline) == 48

    def test_peak_trough_indices(self):
        period = 4
        pattern = [1000.0, 5000.0, 2000.0, 500.0]
        loads = pattern * 5
        result = seasonal_summary(loads, period=period)
        assert result["peak_period_index"] == 1
        assert result["trough_period_index"] == 3

def test_seasonal_baseline_periodicity():
    data = [float(i % 24) for i in range(72)]
    baseline = seasonal_baseline(data, period=24)
    # positions 0, 24, 48 should all have the same baseline
    assert abs(baseline[0] - baseline[24]) < 1e-9
    assert abs(baseline[0] - baseline[48]) < 1e-9


def test_linear_trend_length():
    result = forecast_linear_trend([1.0, 2.0, 3.0, 4.0], horizon=10)
    assert len(result) == 10


def test_linear_trend_direction():
    # Ascending series → future values should be higher than last historical
    result = forecast_linear_trend(list(range(20)), horizon=5)
    assert result[0] > 18.0


def test_detect_spikes_finds_outlier():
    data = [10.0] * 100
    data[50] = 1000.0
    spikes = detect_spikes(data)
    assert 50 in spikes


def test_detect_spikes_empty_on_flat():
    data = [5.0] * 50
    assert detect_spikes(data) == []


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
