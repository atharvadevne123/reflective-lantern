"""Forecaster module tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.forecaster import compute_sma, detect_volume_spike, fit_linear_trend, forecast_volume


def test_compute_sma_basic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compute_sma(values, window=3)
    assert len(result) == 3
    assert result[0] == pytest.approx(2.0)
    assert result[1] == pytest.approx(3.0)
    assert result[2] == pytest.approx(4.0)


def test_compute_sma_window_one():
    values = [10.0, 20.0, 30.0]
    result = compute_sma(values, window=1)
    assert result == pytest.approx([10.0, 20.0, 30.0])


def test_compute_sma_window_equals_length():
    values = [2.0, 4.0, 6.0]
    result = compute_sma(values, window=3)
    assert len(result) == 1
    assert result[0] == pytest.approx(4.0)


def test_compute_sma_window_too_large():
    with pytest.raises(ValueError, match="exceeds series length"):
        compute_sma([1.0, 2.0], window=5)


def test_compute_sma_window_zero():
    with pytest.raises(ValueError, match="window must be"):
        compute_sma([1.0, 2.0, 3.0], window=0)


def test_fit_linear_trend_increasing():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    slope, intercept = fit_linear_trend(values)
    assert slope == pytest.approx(1.0)


def test_fit_linear_trend_flat():
    values = [5.0, 5.0, 5.0, 5.0]
    slope, intercept = fit_linear_trend(values)
    assert slope == pytest.approx(0.0, abs=1e-10)


def test_fit_linear_trend_single_value():
    slope, intercept = fit_linear_trend([42.0])
    assert slope == 0.0
    assert intercept == pytest.approx(42.0)


def test_fit_linear_trend_empty():
    slope, intercept = fit_linear_trend([])
    assert slope == 0.0
    assert intercept == 0.0


def test_forecast_volume_increasing_trend():
    counts = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    result = forecast_volume(counts, horizon_hours=3)
    assert result["trend"] == "increasing"
    assert len(result["forecast"]) == 3
    assert all(v >= 0 for v in result["forecast"])


def test_forecast_volume_stable_trend():
    counts = [10.0] * 10
    result = forecast_volume(counts, horizon_hours=5)
    assert result["trend"] == "stable"


def test_forecast_volume_insufficient_data():
    result = forecast_volume([5.0, 6.0], horizon_hours=4)
    assert result["trend"] == "insufficient_data"
    assert len(result["forecast"]) == 4


def test_forecast_volume_empty():
    result = forecast_volume([], horizon_hours=3)
    assert result["forecast"] == []


def test_forecast_volume_no_negative():
    counts = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    result = forecast_volume(counts, horizon_hours=10)
    assert all(v >= 0 for v in result["forecast"])


def test_forecast_volume_metadata_keys():
    counts = list(range(10, 20))
    result = forecast_volume(counts, horizon_hours=2)
    assert "forecast" in result
    assert "trend" in result
    assert "slope" in result
    assert "n_historical" in result


def test_detect_spike_insufficient():
    result = detect_volume_spike([1.0, 2.0, 3.0])
    assert result["spikes"] == []


def test_detect_spike_normal_distribution():
    counts = [10.0] * 20 + [11.0] * 5
    result = detect_volume_spike(counts)
    assert result["mean"] > 0
    assert result["std"] >= 0


def test_detect_spike_detects_outlier():
    counts = [10.0] * 20 + [100.0]
    result = detect_volume_spike(counts, z_threshold=2.0)
    assert any(s["count"] == 100.0 for s in result["spikes"])


def test_detect_spike_flat_series():
    counts = [5.0] * 10
    result = detect_volume_spike(counts)
    assert result["spikes"] == []
    assert result["std"] == 0.0


def test_detect_spike_z_scores_present():
    counts = [10.0] * 18 + [50.0, 60.0]
    result = detect_volume_spike(counts)
    for spike in result["spikes"]:
        assert "z_score" in spike
        assert "hour_index" in spike
