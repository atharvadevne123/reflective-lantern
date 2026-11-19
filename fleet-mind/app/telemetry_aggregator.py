"""Aggregate raw telemetry readings into summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median


@dataclass
class TelemetryReading:
    vehicle_id: str
    engine_temp_c: float
    oil_pressure_bar: float
    brake_pad_mm: float
    battery_voltage_v: float
    odometer_km: float


def aggregate(readings: list[TelemetryReading]) -> dict[str, float]:
    """Return mean of each numeric field across all readings."""
    if not readings:
        return {}
    fields = ["engine_temp_c", "oil_pressure_bar", "brake_pad_mm", "battery_voltage_v", "odometer_km"]
    return {f: mean(getattr(r, f) for r in readings) for f in fields}


def median_values(readings: list[TelemetryReading]) -> dict[str, float]:
    """Return median of each numeric field."""
    if not readings:
        return {}
    fields = ["engine_temp_c", "oil_pressure_bar", "brake_pad_mm", "battery_voltage_v", "odometer_km"]
    return {f: median(getattr(r, f) for r in readings) for f in fields}


def outlier_vehicles(
    readings: list[TelemetryReading],
    field: str,
    z_threshold: float = 2.0,
) -> list[str]:
    """Return vehicle IDs whose field value is more than z_threshold std devs from mean."""
    values = [getattr(r, field) for r in readings]
    if len(values) < 2:
        return []
    mu = mean(values)
    # manual std to avoid importing stdev from statistics twice
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    sigma = variance ** 0.5
    if sigma == 0:
        return []
    return [r.vehicle_id for r in readings if abs(getattr(r, field) - mu) / sigma > z_threshold]


def group_by_vehicle(readings: list[TelemetryReading]) -> dict[str, list[TelemetryReading]]:
    """Group readings by vehicle_id."""
    result: dict[str, list[TelemetryReading]] = {}
    for r in readings:
        result.setdefault(r.vehicle_id, []).append(r)
    return result


def latest_per_vehicle(readings: list[TelemetryReading]) -> dict[str, TelemetryReading]:
    """Return the last reading for each vehicle (by list order)."""
    result: dict[str, TelemetryReading] = {}
    for r in readings:
        result[r.vehicle_id] = r
    return result
