"""Tests for fleet-mind app/validation_helpers.py."""

from __future__ import annotations

from datetime import date

from app.validation_helpers import (
    all_valid,
    validate_battery_voltage,
    validate_date_range,
    validate_engine_temp,
    validate_odometer,
    validate_vehicle_id,
)


class TestValidateOdometer:
    def test_valid_reading(self) -> None:
        assert validate_odometer(50_000.0) == []

    def test_negative_reading(self) -> None:
        errors = validate_odometer(-1.0)
        assert len(errors) > 0

    def test_less_than_previous(self) -> None:
        errors = validate_odometer(1000.0, previous_km=5000.0)
        assert len(errors) > 0

    def test_excessive_reading(self) -> None:
        errors = validate_odometer(3_000_000.0)
        assert len(errors) > 0


class TestValidateEngineTemp:
    def test_normal_temp(self) -> None:
        assert validate_engine_temp(90.0) == []

    def test_too_cold(self) -> None:
        errors = validate_engine_temp(-50.0)
        assert len(errors) > 0

    def test_too_hot(self) -> None:
        errors = validate_engine_temp(350.0)
        assert len(errors) > 0


class TestValidateBatteryVoltage:
    def test_valid_voltage(self) -> None:
        assert validate_battery_voltage(12.5) == []

    def test_negative_voltage(self) -> None:
        errors = validate_battery_voltage(-1.0)
        assert len(errors) > 0

    def test_excessive_voltage(self) -> None:
        errors = validate_battery_voltage(100.0)
        assert len(errors) > 0


class TestValidateDateRange:
    def test_valid_range(self) -> None:
        assert validate_date_range(date(2026, 1, 1), date(2026, 12, 31)) == []

    def test_end_before_start(self) -> None:
        errors = validate_date_range(date(2026, 12, 1), date(2026, 1, 1))
        assert len(errors) > 0


class TestValidateVehicleId:
    def test_valid_id(self) -> None:
        assert validate_vehicle_id("VH-001") == []

    def test_empty_id(self) -> None:
        errors = validate_vehicle_id("")
        assert len(errors) > 0

    def test_too_long_id(self) -> None:
        errors = validate_vehicle_id("x" * 60)
        assert len(errors) > 0


class TestAllValid:
    def test_all_empty(self) -> None:
        assert all_valid([[], []]) is True

    def test_one_has_error(self) -> None:
        assert all_valid([[], ["error"]]) is False
