"""Extended tests for Fleet-Mind alerting thresholds and drift alerts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.alerting import _CRITICAL_THRESHOLDS, alert_on_drift, check_critical_thresholds


def test_critical_thresholds_dict_not_empty():
    assert len(_CRITICAL_THRESHOLDS) > 0


def test_critical_thresholds_has_engine_temp():
    assert "engine_temp_c" in _CRITICAL_THRESHOLDS


def test_critical_thresholds_has_oil_pressure():
    assert "oil_pressure_bar" in _CRITICAL_THRESHOLDS


def test_safe_readings_produce_no_alerts():
    reading = {
        "engine_temp_c": 90.0,
        "oil_pressure_bar": 3.0,
        "brake_pad_mm": 8.0,
        "battery_voltage_v": 12.5,
    }
    alerts = check_critical_thresholds("v001", reading)
    assert alerts == []


def test_critical_engine_temp_triggers_alert():
    reading = {"engine_temp_c": 120.0}
    alerts = check_critical_thresholds("v001", reading)
    assert any(a["field"] == "engine_temp_c" for a in alerts)


def test_critical_battery_triggers_alert():
    reading = {"battery_voltage_v": 9.5}
    alerts = check_critical_thresholds("v001", reading)
    assert any(a["field"] == "battery_voltage_v" for a in alerts)


def test_alert_dict_has_required_keys():
    reading = {"engine_temp_c": 120.0}
    alerts = check_critical_thresholds("v001", reading)
    assert len(alerts) == 1
    alert = alerts[0]
    for key in ("vehicle_id", "alert_type", "field", "value", "threshold"):
        assert key in alert


def test_alert_vehicle_id_matches():
    reading = {"brake_pad_mm": 1.0}
    alerts = check_critical_thresholds("truck-99", reading)
    if alerts:
        assert alerts[0]["vehicle_id"] == "truck-99"


def test_missing_field_in_reading_skipped():
    alerts = check_critical_thresholds("v001", {"speed_kmh": 60.0})
    assert alerts == []


def test_multiple_critical_readings_produce_multiple_alerts():
    reading = {
        "engine_temp_c": 120.0,
        "oil_pressure_bar": 0.5,
        "brake_pad_mm": 1.0,
    }
    alerts = check_critical_thresholds("v001", reading)
    assert len(alerts) >= 2


def test_alert_on_drift_does_not_raise():
    alert_on_drift("speed_kmh", 0.45, 0.001)


def test_alert_on_drift_emits_warning(caplog):
    with caplog.at_level(logging.WARNING):
        alert_on_drift("engine_temp_c", 0.6, 0.0001)
    assert any("DRIFT ALERT" in r.message for r in caplog.records)


@pytest.mark.parametrize("field,safe_value", [
    ("engine_temp_c", 100.0),
    ("oil_pressure_bar", 2.0),
    ("brake_pad_mm", 5.0),
])
def test_safe_values_do_not_trigger(field, safe_value):
    alerts = check_critical_thresholds("v1", {field: safe_value})
    assert alerts == []
