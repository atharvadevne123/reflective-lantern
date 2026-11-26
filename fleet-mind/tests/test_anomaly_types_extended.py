"""Extended anomaly types tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.anomaly_types import (
    ANOMALY_RULES,
    AnomalyRule,
    classify_anomalies,
)


def test_anomaly_rule_check_above_triggers_when_exceeded() -> None:
    rule = AnomalyRule("test", "desc", "temp", 100.0, "above", "critical")
    assert rule.check({"temp": 101.0}) is True


def test_anomaly_rule_check_above_no_trigger_at_threshold() -> None:
    rule = AnomalyRule("test", "desc", "temp", 100.0, "above", "critical")
    assert rule.check({"temp": 100.0}) is False


def test_anomaly_rule_check_below_triggers_when_under() -> None:
    rule = AnomalyRule("test", "desc", "pressure", 2.0, "below", "warning")
    assert rule.check({"pressure": 1.5}) is True


def test_anomaly_rule_check_below_no_trigger_at_threshold() -> None:
    rule = AnomalyRule("test", "desc", "pressure", 2.0, "below", "warning")
    assert rule.check({"pressure": 2.0}) is False


def test_anomaly_rule_check_missing_field_returns_false() -> None:
    rule = AnomalyRule("test", "desc", "missing_field", 50.0, "above", "info")
    assert rule.check({"other_field": 100.0}) is False


@pytest.mark.parametrize(
    "field,value,direction,threshold,expected",
    [
        ("temp", 110.0, "above", 100.0, True),
        ("temp", 90.0, "above", 100.0, False),
        ("pressure", 1.0, "below", 2.0, True),
        ("pressure", 3.0, "below", 2.0, False),
        ("voltage", 9.5, "below", 10.0, True),
        ("voltage", 12.0, "below", 10.0, False),
    ],
)
def test_anomaly_rule_check_parametrized(
    field: str, value: float, direction: str, threshold: float, expected: bool
) -> None:
    rule = AnomalyRule("r", "d", field, threshold, direction, "warning")
    assert rule.check({field: value}) is expected


def test_classify_anomalies_returns_list() -> None:
    result = classify_anomalies({})
    assert isinstance(result, list)


def test_classify_anomalies_no_anomalies_empty() -> None:
    normal = {
        "engine_temp_c": 85.0,
        "oil_pressure_bar": 3.5,
        "brake_pad_mm": 8.0,
        "battery_voltage_v": 12.6,
        "coolant_temp_c": 90.0,
        "transmission_temp_c": 80.0,
        "tire_pressure_bar": 2.2,
    }
    result = classify_anomalies(normal)
    assert result == []


def test_classify_anomalies_engine_overheat_detected() -> None:
    reading = {"engine_temp_c": 110.0}
    result = classify_anomalies(reading)
    names = [r["name"] for r in result]
    assert "engine_overheat" in names


def test_classify_anomalies_critical_before_warning() -> None:
    reading = {"engine_temp_c": 110.0}
    result = classify_anomalies(reading)
    if len(result) > 1:
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        for i in range(len(result) - 1):
            assert (
                severity_order[result[i]["severity"]] <= severity_order[result[i + 1]["severity"]]
            )


def test_classify_anomalies_result_has_required_keys() -> None:
    reading = {"engine_temp_c": 110.0}
    result = classify_anomalies(reading)
    assert len(result) > 0
    for item in result:
        for key in ("name", "description", "severity", "field", "value", "threshold"):
            assert key in item


def test_classify_anomalies_multiple_anomalies() -> None:
    reading = {
        "engine_temp_c": 110.0,
        "oil_pressure_bar": 0.5,
        "battery_voltage_v": 9.0,
    }
    result = classify_anomalies(reading)
    assert len(result) >= 2


def test_anomaly_rules_non_empty() -> None:
    assert len(ANOMALY_RULES) > 0


@pytest.mark.parametrize("rule", ANOMALY_RULES)
def test_anomaly_rules_valid_direction(rule: AnomalyRule) -> None:
    assert rule.direction in ("above", "below")


@pytest.mark.parametrize("rule", ANOMALY_RULES)
def test_anomaly_rules_valid_severity(rule: AnomalyRule) -> None:
    assert rule.severity in ("critical", "warning", "info")


def test_classify_anomalies_tire_pressure_low() -> None:
    reading = {"tire_pressure_bar": 1.5}
    result = classify_anomalies(reading)
    names = [r["name"] for r in result]
    assert "tire_pressure_low" in names
