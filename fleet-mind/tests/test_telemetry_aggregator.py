"""Tests for fleet-mind app/telemetry_aggregator.py."""

from __future__ import annotations

import pytest

from app.telemetry_aggregator import (
    TelemetryReading,
    aggregate,
    group_by_vehicle,
    latest_per_vehicle,
    median_values,
    outlier_vehicles,
)


def _r(vid: str, **kwargs) -> TelemetryReading:
    defaults = dict(engine_temp_c=90.0, oil_pressure_bar=3.0, brake_pad_mm=8.0, battery_voltage_v=12.5, odometer_km=50_000.0)
    defaults.update(kwargs)
    return TelemetryReading(vehicle_id=vid, **defaults)


class TestAggregate:
    def test_empty_returns_empty(self) -> None:
        assert aggregate([]) == {}

    def test_mean_calculated(self) -> None:
        readings = [_r("v1", engine_temp_c=80.0), _r("v2", engine_temp_c=100.0)]
        result = aggregate(readings)
        assert result["engine_temp_c"] == pytest.approx(90.0)


class TestMedianValues:
    def test_empty_returns_empty(self) -> None:
        assert median_values([]) == {}

    def test_median_of_three(self) -> None:
        readings = [_r("v1", engine_temp_c=80.0), _r("v2", engine_temp_c=90.0), _r("v3", engine_temp_c=100.0)]
        result = median_values(readings)
        assert result["engine_temp_c"] == pytest.approx(90.0)


class TestOutlierVehicles:
    def test_no_outliers_in_uniform_data(self) -> None:
        readings = [_r(f"v{i}", engine_temp_c=90.0) for i in range(5)]
        result = outlier_vehicles(readings, "engine_temp_c")
        assert result == []

    def test_detects_outlier(self) -> None:
        readings = [_r(f"v{i}", engine_temp_c=90.0) for i in range(10)]
        readings.append(_r("outlier", engine_temp_c=200.0))
        result = outlier_vehicles(readings, "engine_temp_c", z_threshold=2.0)
        assert "outlier" in result


class TestGroupByVehicle:
    def test_groups_correctly(self) -> None:
        readings = [_r("v1"), _r("v2"), _r("v1")]
        groups = group_by_vehicle(readings)
        assert len(groups["v1"]) == 2
        assert len(groups["v2"]) == 1


class TestLatestPerVehicle:
    def test_returns_last_reading(self) -> None:
        r1 = _r("v1", engine_temp_c=80.0)
        r2 = _r("v1", engine_temp_c=95.0)
        result = latest_per_vehicle([r1, r2])
        assert result["v1"].engine_temp_c == pytest.approx(95.0)
