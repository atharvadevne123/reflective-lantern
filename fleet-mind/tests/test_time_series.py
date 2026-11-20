"""Tests for time-series rolling statistics and degradation detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.time_series import (
    _VEHICLE_WINDOWS,
    _get_window,
    detect_monotonic_degradation,
    rolling_stats,
    update_window,
)


@pytest.fixture(autouse=True)
def reset_windows():
    _VEHICLE_WINDOWS.clear()
    yield
    _VEHICLE_WINDOWS.clear()


READING = {
    "speed_kmh": 80.0,
    "engine_temp_c": 90.0,
    "oil_pressure_bar": 3.5,
    "fuel_consumption_lph": 8.0,
    "battery_voltage_v": 12.8,
    "brake_pad_mm": 7.0,
    "tire_pressure_bar": 2.3,
    "rpm": 2500.0,
    "coolant_temp_c": 85.0,
    "transmission_temp_c": 75.0,
}


class TestUpdateWindow:
    def test_creates_window(self):
        update_window("VH-TS-01", READING)
        assert "VH-TS-01" in _VEHICLE_WINDOWS

    def test_appends_values(self):
        for _ in range(5):
            update_window("VH-TS-02", READING)
        assert len(_get_window("VH-TS-02", "speed_kmh")) == 5


class TestRollingStats:
    def test_none_when_too_few(self):
        update_window("VH-TS-03", READING)
        assert rolling_stats("VH-TS-03", "engine_temp_c") is None

    def test_returns_dict_with_required_keys(self):
        for _ in range(5):
            update_window("VH-TS-04", READING)
        stats = rolling_stats("VH-TS-04", "engine_temp_c")
        assert stats is not None
        for key in ("mean", "std", "slope"):
            assert key in stats

    def test_constant_series_zero_std(self):
        reading = {**READING, "engine_temp_c": 90.0}
        for _ in range(10):
            update_window("VH-TS-05", reading)
        stats = rolling_stats("VH-TS-05", "engine_temp_c")
        assert stats is not None
        assert abs(stats["std"]) < 1e-9


class TestDetectMonotonicDegradation:
    def test_flat_trend_not_degrading(self):
        for _ in range(10):
            update_window("VH-TS-06", READING)
        assert detect_monotonic_degradation("VH-TS-06", "engine_temp_c") is False

    def test_rising_brake_temp_flagged(self):
        for i in range(10):
            r = {**READING, "engine_temp_c": 80.0 + i * 3.0}
            update_window("VH-TS-07", r)
        assert detect_monotonic_degradation("VH-TS-07", "engine_temp_c", min_slope=0.05) is True

    @pytest.mark.parametrize("slope_threshold", [0.01, 0.1, 0.5])
    def test_parametrized_slope_threshold(self, slope_threshold):
        for i in range(10):
            r = {**READING, "oil_pressure_bar": 5.0 - i * 0.1}
            update_window("VH-TS-08", r)
        result = detect_monotonic_degradation("VH-TS-08", "oil_pressure_bar", min_slope=slope_threshold)
        assert isinstance(result, bool)
