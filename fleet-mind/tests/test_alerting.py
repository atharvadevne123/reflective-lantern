"""Tests for Fleet-Mind alerting utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.alerting import alert_on_drift, check_critical_thresholds


def _safe_reading() -> dict:
    return {
        "engine_temp_c": 90.0,
        "oil_pressure_bar": 3.0,
        "brake_pad_mm": 5.0,
        "battery_voltage_v": 12.6,
        "coolant_temp_c": 85.0,
        "transmission_temp_c": 70.0,
    }


def test_no_alerts_for_safe_reading():
    alerts = check_critical_thresholds("VH-001", _safe_reading())
    assert alerts == []


def test_critical_engine_temp_raises_alert():
    reading = _safe_reading()
    reading["engine_temp_c"] = 115.0  # exactly at threshold
    alerts = check_critical_thresholds("VH-001", reading)
    alert_types = [a["alert_type"] for a in alerts]
    assert "CRITICAL_ENGINE_OVERHEAT" in alert_types


def test_critical_oil_pressure_raises_alert():
    reading = _safe_reading()
    reading["oil_pressure_bar"] = 0.5  # below threshold
    alerts = check_critical_thresholds("VH-001", reading)
    alert_types = [a["alert_type"] for a in alerts]
    assert "CRITICAL_OIL_PRESSURE_LOW" in alert_types


def test_critical_brake_raises_alert():
    reading = _safe_reading()
    reading["brake_pad_mm"] = 1.0  # below threshold
    alerts = check_critical_thresholds("VH-001", reading)
    alert_types = [a["alert_type"] for a in alerts]
    assert "CRITICAL_BRAKE_WORN" in alert_types


def test_alert_includes_vehicle_id():
    reading = _safe_reading()
    reading["engine_temp_c"] = 120.0
    alerts = check_critical_thresholds("VH-XYZ", reading)
    assert all(a["vehicle_id"] == "VH-XYZ" for a in alerts)


def test_alert_includes_field_and_value():
    reading = _safe_reading()
    reading["battery_voltage_v"] = 9.0
    alerts = check_critical_thresholds("VH-001", reading)
    battery_alerts = [a for a in alerts if a["field"] == "battery_voltage_v"]
    assert len(battery_alerts) == 1
    assert battery_alerts[0]["value"] == 9.0


def test_missing_fields_do_not_raise_alert():
    alerts = check_critical_thresholds("VH-001", {})
    assert alerts == []


def test_multiple_critical_readings_produce_multiple_alerts():
    reading = {
        "engine_temp_c": 120.0,
        "oil_pressure_bar": 0.5,
        "brake_pad_mm": 1.0,
    }
    alerts = check_critical_thresholds("VH-001", reading)
    assert len(alerts) == 3


def test_alert_on_drift_does_not_raise():
    # Should log a warning without raising
    alert_on_drift("speed_kmh", 0.45, 0.001)


@pytest.mark.parametrize(
    "field,safe_val,critical_val",
    [
        ("engine_temp_c", 90.0, 120.0),
        ("oil_pressure_bar", 3.0, 0.5),
        ("brake_pad_mm", 8.0, 1.0),
    ],
)
def test_threshold_boundary(field, safe_val, critical_val):
    safe = _safe_reading()
    safe[field] = safe_val
    assert check_critical_thresholds("V", safe) == []

    critical = _safe_reading()
    critical[field] = critical_val
    alerts = check_critical_thresholds("V", critical)
    assert len(alerts) >= 1
