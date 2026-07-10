"""Tests for time-series analysis utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.time_series import compute_trend, detect_load_spikes, seasonal_summary


class TestComputeTrend:
    def test_increasing_trend(self):
        loads = [float(1000 + i * 50) for i in range(50)]
        result = compute_trend(loads, window=10)
        assert result["direction"] == "increasing"
        assert result["slope_mw_per_hour"] > 0

    def test_decreasing_trend(self):
        loads = [float(5000 - i * 50) for i in range(50)]
        result = compute_trend(loads, window=10)
        assert result["direction"] == "decreasing"

    def test_stable_trend(self):
        loads = [3000.0] * 48
        result = compute_trend(loads, window=24)
        assert result["direction"] == "stable"
        assert result["slope_mw_per_hour"] == pytest.approx(0.0, abs=1.0)

    def test_insufficient_data(self):
        result = compute_trend([3000.0, 3100.0], window=24)
        assert result["direction"] == "unknown"
        assert result["trend"] is None

    def test_returns_expected_keys(self):
        loads = [3000.0 + i for i in range(30)]
        result = compute_trend(loads, window=10)
        assert "sma_last" in result
        assert "slope_mw_per_hour" in result
        assert "direction" in result

    @pytest.mark.parametrize("window", [6, 12, 24])
    def test_various_window_sizes(self, window):
        loads = [3000.0 + i * 10 for i in range(60)]
        result = compute_trend(loads, window=window)
        assert result["window_size"] == window
        assert result["sma_last"] is not None


class TestDetectLoadSpikes:
    def test_detects_obvious_spike(self):
        loads = [3000.0] * 50 + [9999.0] + [3000.0] * 50
        spikes = detect_load_spikes(loads)
        assert len(spikes) >= 1
        assert any(s["index"] == 50 for s in spikes)

    def test_no_spikes_uniform(self):
        loads = [3000.0] * 100
        spikes = detect_load_spikes(loads)
        assert len(spikes) == 0

    def test_returns_expected_keys(self):
        loads = [3000.0] * 50 + [9000.0] + [3000.0] * 50
        spikes = detect_load_spikes(loads)
        if spikes:
            assert "index" in spikes[0]
            assert "load_mw" in spikes[0]
            assert "z_score" in spikes[0]
            assert "deviation_mw" in spikes[0]

    def test_insufficient_data(self):
        spikes = detect_load_spikes([3000.0, 3100.0])
        assert spikes == []


class TestSeasonalSummary:
    def test_computes_seasonal_means(self):
        period = 24
        pattern = [float(1000 + i * 100) for i in range(period)]
        loads = pattern * 3
        result = seasonal_summary(loads, period=period)
        assert len(result["seasonal_means"]) == period
        assert result["period"] == period

    def test_peak_trough_indices(self):
        period = 4
        pattern = [1000.0, 5000.0, 2000.0, 500.0]
        loads = pattern * 5
        result = seasonal_summary(loads, period=period)
        assert result["peak_period_index"] == 1
        assert result["trough_period_index"] == 3

    def test_insufficient_data(self):
        result = seasonal_summary([3000.0] * 5, period=24)
        assert result["seasonal_means"] == []
