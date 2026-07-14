"""Time-series forecasting utility tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.time_series import (
    cumulative_sum,
    daily_totals,
    detect_spikes,
    first_nonzero,
    forecast_linear_trend,
    moving_max,
    normalize_series,
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

def test_cumulative_sum_basic() -> None:
    assert cumulative_sum([1.0, 2.0, 3.0]) == [1.0, 3.0, 6.0]


def test_cumulative_sum_empty() -> None:
    assert cumulative_sum([]) == []


def test_moving_max_basic() -> None:
    import math
    result = moving_max([1.0, 3.0, 2.0, 5.0], window=3)
    assert math.isnan(result[0])
    assert result[2] == pytest.approx(3.0)
    assert result[3] == pytest.approx(5.0)


def test_moving_max_too_short_all_nan() -> None:
    import math
    result = moving_max([1.0, 2.0], window=5)
    assert all(math.isnan(v) for v in result)


def test_normalize_series_basic() -> None:
    result = normalize_series([0.0, 5.0, 10.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_series_constant() -> None:
    assert normalize_series([7.0, 7.0, 7.0]) == [0.0, 0.0, 0.0]


def test_daily_totals_default_period() -> None:
    result = daily_totals([1.0] * 48, period=24)
    assert result == [24.0, 24.0]


def test_daily_totals_empty() -> None:
    assert daily_totals([]) == []


def test_first_nonzero_finds_index() -> None:
    assert first_nonzero([0.0, 0.0, 3.0]) == 2


def test_first_nonzero_all_zeros() -> None:
    assert first_nonzero([0.0, 0.0]) == -1


def test_first_nonzero_empty() -> None:
    assert first_nonzero([]) == -1
