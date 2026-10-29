"""Tests for multi-label anomaly classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.anomaly_types import AnomalyRule, classify_anomalies


@pytest.fixture
def healthy_reading():
    return {
        "speed_kmh": 80.0,
        "engine_temp_c": 88.0,
        "oil_pressure_bar": 3.5,
        "fuel_consumption_lph": 7.5,
        "battery_voltage_v": 12.8,
        "brake_pad_mm": 6.0,
        "tire_pressure_bar": 2.3,
        "rpm": 2400.0,
        "coolant_temp_c": 84.0,
        "transmission_temp_c": 72.0,
    }


class TestAnomalyRule:
    def test_above_triggers_when_greater(self):
        rule = AnomalyRule("test", "test", "engine_temp_c", 100.0, "above", "warning")
        assert rule.check({"engine_temp_c": 105.0}) is True

    def test_above_does_not_trigger_when_equal(self):
        rule = AnomalyRule("test", "test", "engine_temp_c", 100.0, "above", "warning")
        assert rule.check({"engine_temp_c": 100.0}) is False

    def test_below_triggers_when_less(self):
        rule = AnomalyRule("test", "test", "oil_pressure_bar", 2.0, "below", "warning")
        assert rule.check({"oil_pressure_bar": 1.5}) is True

    def test_missing_field_returns_false(self):
        rule = AnomalyRule("test", "test", "engine_temp_c", 100.0, "above", "warning")
        assert rule.check({}) is False


class TestClassifyAnomalies:
    def test_healthy_reading_no_anomalies(self, healthy_reading):
        anomalies = classify_anomalies(healthy_reading)
        critical = [a for a in anomalies if a["severity"] == "critical"]
        assert len(critical) == 0

    def test_overheated_engine_flagged(self, healthy_reading):
        reading = {**healthy_reading, "engine_temp_c": 110.0}
        anomalies = classify_anomalies(reading)
        names = [a["name"] for a in anomalies]
        assert "engine_overheat" in names

    def test_multiple_anomalies_detected(self, healthy_reading):
        reading = {
            **healthy_reading,
            "engine_temp_c": 110.0,
            "oil_pressure_bar": 0.5,
            "brake_pad_mm": 1.0,
        }
        anomalies = classify_anomalies(reading)
        assert len(anomalies) >= 3

    def test_critical_before_warning(self, healthy_reading):
        reading = {**healthy_reading, "engine_temp_c": 110.0, "oil_pressure_bar": 1.5}
        anomalies = classify_anomalies(reading)
        if len(anomalies) >= 2:
            severities = [a["severity"] for a in anomalies]
            critical_indices = [i for i, s in enumerate(severities) if s == "critical"]
            warning_indices = [i for i, s in enumerate(severities) if s == "warning"]
            if critical_indices and warning_indices:
                assert max(critical_indices) < max(warning_indices)

    @pytest.mark.parametrize("field,value,should_trigger", [
        ("brake_pad_mm", 1.0, True),
        ("brake_pad_mm", 8.0, False),
        ("battery_voltage_v", 10.0, True),
        ("tire_pressure_bar", 1.5, True),
    ])
    def test_parametrized_field_values(self, healthy_reading, field, value, should_trigger):
        reading = {**healthy_reading, field: value}
        anomalies = classify_anomalies(reading)
        has_anomaly = any(a["field"] == field for a in anomalies)
        assert has_anomaly == should_trigger

    def test_anomaly_dict_has_required_keys(self, healthy_reading):
        reading = {**healthy_reading, "engine_temp_c": 120.0}
        anomalies = classify_anomalies(reading)
        if anomalies:
            for key in ("name", "severity", "field", "value", "threshold"):
                assert key in anomalies[0]
