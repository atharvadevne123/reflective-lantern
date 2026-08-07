"""Tests for domain-specific sensor validators."""

from __future__ import annotations

import math

import pytest

VALID = {
    "temperature": 75.0,
    "pressure": 50.0,
    "vibration": 2.1,
    "cycle_time": 28.0,
    "tool_wear": 15.0,
    "power_consumption": 98.0,
    "humidity": 45.0,
}


def test_valid_reading_passes():
    from app.validators import validate_sensor_reading

    assert validate_sensor_reading(VALID) == []


def test_is_valid_returns_true_for_valid():
    from app.validators import is_valid_sensor_reading

    assert is_valid_sensor_reading(VALID) is True


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("temperature", 999.0),
        ("temperature", -100.0),
        ("pressure", -1.0),
        ("pressure", 300.0),
        ("vibration", -0.1),
        ("vibration", 100.0),
        ("cycle_time", 0.0),
        ("cycle_time", 10000.0),
        ("tool_wear", -1.0),
        ("humidity", -5.0),
        ("humidity", 150.0),
    ],
)
def test_out_of_range_produces_error(field: str, bad_value: float):
    from app.validators import validate_sensor_reading

    errors = validate_sensor_reading({**VALID, field: bad_value})
    assert any(field in e for e in errors)


def test_missing_field_produces_error():
    from app.validators import validate_sensor_reading

    incomplete = {k: v for k, v in VALID.items() if k != "humidity"}
    errors = validate_sensor_reading(incomplete)
    assert any("humidity" in e for e in errors)


def test_nan_value_rejected():
    from app.validators import validate_sensor_reading

    errors = validate_sensor_reading({**VALID, "temperature": math.nan})
    assert errors


def test_inf_value_rejected():
    from app.validators import validate_sensor_reading

    errors = validate_sensor_reading({**VALID, "pressure": math.inf})
    assert errors


def test_sanitize_fills_missing_with_median():
    from app.validators import sanitize_sensor_reading

    incomplete: dict = {k: v for k, v in VALID.items() if k != "humidity"}
    result = sanitize_sensor_reading(incomplete)
    assert "humidity" in result
    assert result["humidity"] == 50.0


def test_sanitize_clips_to_range():
    from app.validators import sanitize_sensor_reading

    result = sanitize_sensor_reading({**VALID, "humidity": 200.0})
    assert result["humidity"] == 100.0


def test_sanitize_handles_non_numeric():
    from app.validators import sanitize_sensor_reading

    result = sanitize_sensor_reading({**VALID, "temperature": "bad"})
    assert result["temperature"] == 75.0


def test_validate_returns_empty_list_for_boundary_low():
    from app.validators import validate_sensor_reading

    reading = {**VALID, "temperature": -50.0, "pressure": 0.0, "humidity": 0.0}
    errors = validate_sensor_reading(reading)
    assert errors == []


def test_validate_returns_errors_for_all_bad_fields():
    from app.validators import validate_sensor_reading

    bad = dict.fromkeys(VALID, -9999.0)
    errors = validate_sensor_reading(bad)
    assert len(errors) >= len(VALID)


def test_is_valid_false_for_nan():
    from app.validators import is_valid_sensor_reading

    assert is_valid_sensor_reading({**VALID, "vibration": math.nan}) is False


def test_sanitize_nan_replaced_with_median():
    from app.validators import sanitize_sensor_reading

    result = sanitize_sensor_reading({**VALID, "temperature": math.nan})
    assert math.isfinite(result["temperature"])


def test_sanitize_inf_clipped():
    from app.validators import sanitize_sensor_reading

    result = sanitize_sensor_reading({**VALID, "tool_wear": math.inf})
    assert math.isfinite(result["tool_wear"])
