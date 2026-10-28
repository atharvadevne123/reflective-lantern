"""Tests for fleet telemetry plausibility validators."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.validators import ValidationIssue, check_critical_thresholds, has_critical_issues, validate_reading


@pytest.fixture
def healthy_reading():
    return {
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


class TestValidateReading:
    def test_healthy_reading_no_issues(self, healthy_reading):
        issues = validate_reading(healthy_reading)
        assert issues == []

    def test_overheating_engine_flagged(self, healthy_reading):
        reading = {**healthy_reading, "engine_temp_c": 170.0}
        issues = validate_reading(reading)
        assert any(i.field == "engine_temp_c" for i in issues)

    @pytest.mark.parametrize("brake_mm,expected_severity", [
        (0.5, "error"),
        (3.5, "warning"),
    ])
    def test_brake_pad_severity(self, healthy_reading, brake_mm, expected_severity):
        reading = {**healthy_reading, "brake_pad_mm": brake_mm}
        issues = validate_reading(reading)
        brake_issues = [i for i in issues if i.field == "brake_pad_mm"]
        if expected_severity == "error":
            assert any(i.severity == "error" for i in brake_issues)


class TestHasCriticalIssues:
    def test_no_issues_returns_false(self):
        assert has_critical_issues([]) is False

    def test_warning_only_returns_false(self):
        issues = [ValidationIssue("engine_temp_c", "msg", "warning")]
        assert has_critical_issues(issues) is False

    def test_error_returns_true(self):
        issues = [ValidationIssue("brake_pad_mm", "msg", "error")]
        assert has_critical_issues(issues) is True


class TestCheckCriticalThresholds:
    def test_normal_reading_no_alerts(self, healthy_reading):
        alerts = check_critical_thresholds("VH-001", healthy_reading)
        assert alerts == []

    def test_overheated_engine_alert(self, healthy_reading):
        reading = {**healthy_reading, "engine_temp_c": 50.0}
        alerts = check_critical_thresholds("VH-001", reading)
        assert any(a["alert_type"] == "CRITICAL_ENGINE_OVERHEAT" for a in alerts)

    def test_alert_contains_vehicle_id(self, healthy_reading):
        reading = {**healthy_reading, "brake_pad_mm": 0.5}
        alerts = check_critical_thresholds("VH-ALERT-01", reading)
        if alerts:
            assert all(a["vehicle_id"] == "VH-ALERT-01" for a in alerts)
