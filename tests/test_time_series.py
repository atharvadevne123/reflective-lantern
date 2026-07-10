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
