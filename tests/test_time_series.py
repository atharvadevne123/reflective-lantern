"""Time-series forecasting utility tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.time_series import (
    detect_spikes,
    exponential_moving_average,
    forecast_linear_trend,
    peak_hours,
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


@pytest.mark.parametrize("window", [1, 3, 7, 24])
def test_sma_various_windows(window):
    data = list(np.random.default_rng(42).uniform(5, 30, 100))
    result = simple_moving_average(data, window=window)
    assert len(result) == 100


def test_detect_spikes_empty_input():
    assert detect_spikes([]) == []


def test_detect_spikes_returns_indices():
    data = [1.0] * 50 + [999.0]
    spikes = detect_spikes(data)
    assert 50 in spikes


def test_peak_hours_returns_top_n():
    data = [1.0, 5.0, 3.0, 9.0, 2.0]
    peaks = peak_hours(data, top_n=2)
    assert len(peaks) == 2
    assert peaks[0] == 3  # index of 9.0


def test_peak_hours_empty_input():
    assert peak_hours([]) == []


def test_peak_hours_top_n_larger_than_series():
    data = [1.0, 2.0]
    assert len(peak_hours(data, top_n=100)) == 2


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_linear_trend_various_horizons(horizon):
    result = forecast_linear_trend(list(range(30)), horizon=horizon)
    assert len(result) == horizon


def test_cumulative_consumption_empty():
    from app.time_series import cumulative_consumption

    assert cumulative_consumption([]) == []


def test_cumulative_consumption_values():
    from app.time_series import cumulative_consumption

    result = cumulative_consumption([1.0, 2.0, 3.0])
    assert result == pytest.approx([1.0, 3.0, 6.0])


def test_cumulative_consumption_monotone():
    from app.time_series import cumulative_consumption

    data = [0.5, 1.5, 2.0, 0.1]
    result = cumulative_consumption(data)
    assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def test_resample_hourly_to_daily_exact():
    from app.time_series import resample_hourly_to_daily

    data = [1.0] * 48  # 2 full days
    result = resample_hourly_to_daily(data)
    assert result == pytest.approx([24.0, 24.0])


def test_resample_hourly_to_daily_partial_day():
    from app.time_series import resample_hourly_to_daily

    data = [1.0] * 25  # 1 day + 1 hour
    result = resample_hourly_to_daily(data)
    assert len(result) == 2
    assert result[0] == pytest.approx(24.0)
    assert result[1] == pytest.approx(1.0)


def test_resample_hourly_to_daily_empty():
    from app.time_series import resample_hourly_to_daily

    assert resample_hourly_to_daily([]) == []


def test_ema_length():
    result = exponential_moving_average([1.0] * 10, alpha=0.3)
    assert len(result) == 10


def test_ema_first_value_equals_input():
    data = [5.0, 10.0, 15.0]
    result = exponential_moving_average(data, alpha=0.5)
    assert result[0] == pytest.approx(5.0)


def test_ema_alpha_one_equals_input():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = exponential_moving_average(data, alpha=1.0)
    assert result == pytest.approx(data)


def test_ema_empty_input():
    assert exponential_moving_average([], alpha=0.3) == []


def test_ema_invalid_alpha_raises():
    with pytest.raises(ValueError):
        exponential_moving_average([1.0, 2.0], alpha=0.0)
    with pytest.raises(ValueError):
        exponential_moving_average([1.0, 2.0], alpha=1.5)


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9, 1.0])
def test_ema_various_alphas(alpha):
    data = [float(i) for i in range(20)]
    result = exponential_moving_average(data, alpha=alpha)
    assert len(result) == 20


def test_forecast_trend_with_seasonality_length():
    from app.time_series import forecast_trend_with_seasonality

    data = [float(i % 24) for i in range(72)]
    result = forecast_trend_with_seasonality(data, horizon=12, period=24)
    assert len(result) == 12


def test_forecast_trend_with_seasonality_non_negative():
    from app.time_series import forecast_trend_with_seasonality

    data = [10.0 + i * 0.1 for i in range(48)]
    result = forecast_trend_with_seasonality(data, horizon=24, period=24)
    assert all(v >= 0 for v in result)


def test_forecast_trend_with_seasonality_empty_input():
    from app.time_series import forecast_trend_with_seasonality

    assert forecast_trend_with_seasonality([], horizon=10) == []


def test_forecast_trend_with_seasonality_zero_horizon():
    from app.time_series import forecast_trend_with_seasonality

    data = list(range(48))
    assert forecast_trend_with_seasonality(data, horizon=0) == []


@pytest.mark.parametrize("horizon,period", [(6, 12), (24, 24), (48, 24), (7, 7)])
def test_forecast_trend_with_seasonality_parametrized(horizon, period):
    from app.time_series import forecast_trend_with_seasonality

    data = [float(i % period) * 2 + 5 for i in range(4 * period)]
    result = forecast_trend_with_seasonality(data, horizon=horizon, period=period)
    assert len(result) == horizon


def test_moving_range_basic():
    from app.time_series import moving_range

    result = moving_range([1.0, 3.0, 2.0, 5.0])
    assert result == pytest.approx([2.0, 1.0, 3.0])


def test_moving_range_empty_input():
    from app.time_series import moving_range

    assert moving_range([]) == []


def test_moving_range_single_element():
    from app.time_series import moving_range

    assert moving_range([5.0]) == []


def test_moving_range_flat_series():
    from app.time_series import moving_range

    result = moving_range([4.0] * 10)
    assert all(v == pytest.approx(0.0) for v in result)


def test_moving_range_length():
    from app.time_series import moving_range

    data = list(range(20))
    assert len(moving_range(data)) == 19


def test_consumption_variance_flat():
    from app.time_series import consumption_variance

    assert consumption_variance([5.0] * 10) == pytest.approx(0.0)


def test_consumption_variance_two_values():
    from app.time_series import consumption_variance

    result = consumption_variance([0.0, 2.0])
    assert result == pytest.approx(1.0)


def test_consumption_variance_empty():
    from app.time_series import consumption_variance

    assert consumption_variance([]) == 0.0


def test_consumption_variance_single():
    from app.time_series import consumption_variance

    assert consumption_variance([7.0]) == 0.0


@pytest.mark.parametrize(
    "data,expected_len",
    [
        ([1.0, 2.0, 3.0], 2),
        ([10.0] * 5, 4),
    ],
)
def test_moving_range_parametrized_length(data, expected_len):
    from app.time_series import moving_range

    assert len(moving_range(data)) == expected_len


# --- New tests for recently added functions ---

def test_moving_range_basic_v2():
    from app.time_series import moving_range
    values = [1.0, 3.0, 2.0, 5.0]
    result = moving_range(values)
    assert result == [2.0, 1.0, 3.0]


def test_moving_range_too_short_v2():
    from app.time_series import moving_range
    assert moving_range([5.0]) == []
    assert moving_range([]) == []


def test_consumption_variance_flat_v2():
    from app.time_series import consumption_variance
    result = consumption_variance([3.0] * 10)
    assert result == 0.0


def test_consumption_variance_known_v2():
    from app.time_series import consumption_variance
    result = consumption_variance([2.0, 4.0])
    assert abs(result - 1.0) < 1e-9


def test_consumption_variance_too_short_v2():
    from app.time_series import consumption_variance
    assert consumption_variance([5.0]) == 0.0


def test_forecast_trend_with_seasonality_length_v2():
    from app.time_series import forecast_trend_with_seasonality
    values = [float(i % 24) for i in range(48)]
    result = forecast_trend_with_seasonality(values, horizon=12, period=24)
    assert len(result) == 12


def test_forecast_trend_with_seasonality_empty_v2():
    from app.time_series import forecast_trend_with_seasonality
    assert forecast_trend_with_seasonality([], horizon=5) == []


def test_forecast_trend_with_seasonality_non_negative_v2():
    from app.time_series import forecast_trend_with_seasonality
    values = [max(0, float(i) + 5) for i in range(48)]
    result = forecast_trend_with_seasonality(values, horizon=12)
    assert all(v >= 0 for v in result)


def test_resample_hourly_to_daily_full_day_v2():
    from app.time_series import resample_hourly_to_daily
    hourly = [1.0] * 48
    result = resample_hourly_to_daily(hourly)
    assert len(result) == 2
    assert all(abs(v - 24.0) < 1e-9 for v in result)


def test_resample_hourly_to_daily_partial_v2():
    from app.time_series import resample_hourly_to_daily
    hourly = [1.0] * 25
    result = resample_hourly_to_daily(hourly)
    assert len(result) == 2


def test_resample_hourly_to_daily_empty_v2():
    from app.time_series import resample_hourly_to_daily
    assert resample_hourly_to_daily([]) == []


def test_cumulative_consumption_basic_v2():
    from app.time_series import cumulative_consumption
    result = cumulative_consumption([1.0, 2.0, 3.0])
    assert abs(result[-1] - 6.0) < 1e-6


def test_cumulative_consumption_empty_v2():
    from app.time_series import cumulative_consumption
    assert cumulative_consumption([]) == []


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_ema_same_length(alpha):
    values = [float(i) for i in range(20)]
    from app.time_series import exponential_moving_average
    result = exponential_moving_average(values, alpha=alpha)
    assert len(result) == len(values)


def test_detect_plateau_flat_series():
    from app.time_series import detect_plateau
    values = [10.0] * 8
    plateaus = detect_plateau(values, tolerance=0.1)
    assert len(plateaus) >= 1
    assert plateaus[0][0] == 0
    assert plateaus[0][1] == 7


def test_detect_plateau_no_plateau():
    from app.time_series import detect_plateau
    values = [1.0, 5.0, 1.0, 5.0, 1.0]
    plateaus = detect_plateau(values, tolerance=0.1)
    assert len(plateaus) == 0


def test_detect_plateau_empty():
    from app.time_series import detect_plateau
    assert detect_plateau([]) == []


def test_clip_outliers_basic():
    from app.time_series import clip_outliers
    values = [1.0] * 8 + [1000.0]
    clipped = clip_outliers(values, upper_pct=90.0)
    assert max(clipped) < 1000.0


def test_clip_outliers_preserves_length():
    from app.time_series import clip_outliers
    values = list(range(20))
    clipped = clip_outliers([float(v) for v in values])
    assert len(clipped) == 20


def test_clip_outliers_empty():
    from app.time_series import clip_outliers
    assert clip_outliers([]) == []


@pytest.mark.parametrize("upper_pct", [75.0, 90.0, 99.0])
def test_clip_outliers_various_percentiles(upper_pct):
    from app.time_series import clip_outliers
    values = list(range(1, 101))
    clipped = clip_outliers([float(v) for v in values], upper_pct=upper_pct)
    assert max(clipped) <= upper_pct + 1
