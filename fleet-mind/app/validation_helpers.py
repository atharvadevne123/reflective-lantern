"""Input validation helpers for fleet telemetry and maintenance data."""

from __future__ import annotations

from datetime import date


def validate_odometer(odometer_km: float, previous_km: float = 0.0) -> list[str]:
    """Return validation errors for an odometer reading."""
    errors: list[str] = []
    if odometer_km < 0:
        errors.append("odometer_km must be non-negative")
    if odometer_km < previous_km:
        errors.append(f"odometer_km ({odometer_km}) is less than previous reading ({previous_km})")
    if odometer_km > 2_000_000:
        errors.append("odometer_km exceeds plausible maximum (2,000,000)")
    return errors


def validate_engine_temp(temp_c: float) -> list[str]:
    """Return validation errors for an engine temperature reading."""
    errors: list[str] = []
    if temp_c < -40:
        errors.append(f"engine_temp_c ({temp_c}) below plausible minimum (-40°C)")
    if temp_c > 300:
        errors.append(f"engine_temp_c ({temp_c}) above plausible maximum (300°C)")
    return errors


def validate_battery_voltage(voltage_v: float) -> list[str]:
    """Return validation errors for a battery voltage reading."""
    errors: list[str] = []
    if voltage_v < 0:
        errors.append("battery_voltage_v must be non-negative")
    if voltage_v > 48:
        errors.append(f"battery_voltage_v ({voltage_v}) exceeds typical max (48V)")
    return errors


def validate_date_range(start: date, end: date) -> list[str]:
    """Return validation errors for a date range."""
    errors: list[str] = []
    if end < start:
        errors.append(f"end date ({end}) is before start date ({start})")
    if (end - start).days > 365 * 5:
        errors.append("date range exceeds 5 years")
    return errors


def validate_vehicle_id(vehicle_id: str) -> list[str]:
    """Return validation errors for a vehicle identifier."""
    errors: list[str] = []
    if not vehicle_id or not vehicle_id.strip():
        errors.append("vehicle_id must not be empty")
    if len(vehicle_id) > 50:
        errors.append("vehicle_id exceeds maximum length (50)")
    return errors


def all_valid(error_lists: list[list[str]]) -> bool:
    """Return True when all error lists are empty."""
    return all(not errs for errs in error_lists)
