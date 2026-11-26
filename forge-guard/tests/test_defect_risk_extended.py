"""Extended defect risk index tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.features import defect_risk_index, sensor_range_feature


@pytest.mark.parametrize(
    "temperature,vibration,tool_wear,pressure,expected",
    [
        (70.0, 3.0, 20.0, 50.0, 0.0),
        (90.0, 3.0, 20.0, 50.0, 0.3),
        (70.0, 6.0, 20.0, 50.0, 0.25),
        (70.0, 3.0, 50.0, 50.0, 0.25),
        (70.0, 3.0, 20.0, 60.0, 0.15),
        (90.0, 6.0, 50.0, 60.0, 0.95),
    ],
)
def test_defect_risk_index_parametrized(
    temperature: float, vibration: float, tool_wear: float, pressure: float, expected: float
) -> None:
    result = defect_risk_index(temperature, vibration, tool_wear, pressure)
    assert result == pytest.approx(expected, abs=1e-5)


def test_defect_risk_index_all_safe_returns_zero() -> None:
    result = defect_risk_index(70.0, 3.0, 20.0, 50.0)
    assert result == 0.0


def test_defect_risk_index_all_critical_capped_at_one() -> None:
    result = defect_risk_index(90.0, 6.0, 50.0, 60.0)
    assert result <= 1.0


def test_defect_risk_index_in_zero_one() -> None:
    result = defect_risk_index(90.0, 6.0, 50.0, 60.0)
    assert 0.0 <= result <= 1.0


@pytest.mark.parametrize("temp", [85.1, 90.0, 100.0, 120.0])
def test_defect_risk_index_temperature_above_threshold(temp: float) -> None:
    result = defect_risk_index(temp, 3.0, 20.0, 50.0)
    assert result >= 0.3


@pytest.mark.parametrize("temp", [0.0, 50.0, 84.9])
def test_defect_risk_index_temperature_below_threshold(temp: float) -> None:
    result_above = defect_risk_index(90.0, 3.0, 20.0, 50.0)
    result_below = defect_risk_index(temp, 3.0, 20.0, 50.0)
    assert result_below < result_above


def test_sensor_range_feature_empty_returns_zeros() -> None:
    result = sensor_range_feature({})
    assert result == {"sensor_min": 0.0, "sensor_max": 0.0, "sensor_range": 0.0}


def test_sensor_range_feature_single_value_no_range() -> None:
    result = sensor_range_feature({"only": 42.0})
    assert result["sensor_range"] == 0.0
    assert result["sensor_min"] == pytest.approx(42.0)
    assert result["sensor_max"] == pytest.approx(42.0)


def test_sensor_range_feature_computes_range() -> None:
    result = sensor_range_feature({"a": 1.0, "b": 5.0, "c": 3.0})
    assert result["sensor_min"] == pytest.approx(1.0)
    assert result["sensor_max"] == pytest.approx(5.0)
    assert result["sensor_range"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "values,expected_range",
    [
        ([10.0, 10.0, 10.0], 0.0),
        ([0.0, 100.0], 100.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 4.0),
    ],
)
def test_sensor_range_feature_various(values: list, expected_range: float) -> None:
    row = {f"s{i}": v for i, v in enumerate(values)}
    result = sensor_range_feature(row)
    assert result["sensor_range"] == pytest.approx(expected_range)


def test_sensor_range_feature_negative_values() -> None:
    result = sensor_range_feature({"a": -5.0, "b": 5.0})
    assert result["sensor_min"] == pytest.approx(-5.0)
    assert result["sensor_max"] == pytest.approx(5.0)
    assert result["sensor_range"] == pytest.approx(10.0)
