"""Extended tests for forge-guard app/features.py defect_risk_index and sensor_range_feature."""

from __future__ import annotations

import pytest

from app.features import defect_risk_index, sensor_range_feature


class TestDefectRiskIndex:
    def test_zero_on_safe_readings(self) -> None:
        score = defect_risk_index(temperature=50.0, vibration=1.0, tool_wear=10.0, pressure=30.0)
        assert score == pytest.approx(0.0)

    def test_high_risk_all_thresholds_exceeded(self) -> None:
        score = defect_risk_index(temperature=90.0, vibration=6.0, tool_wear=50.0, pressure=60.0)
        assert score > 0.9

    def test_capped_at_one(self) -> None:
        score = defect_risk_index(temperature=200.0, vibration=100.0, tool_wear=100.0, pressure=200.0)
        assert score <= 1.0

    def test_temperature_alone(self) -> None:
        score = defect_risk_index(temperature=90.0, vibration=1.0, tool_wear=10.0, pressure=30.0)
        assert score == pytest.approx(0.30)

    def test_vibration_alone(self) -> None:
        score = defect_risk_index(temperature=50.0, vibration=6.0, tool_wear=10.0, pressure=30.0)
        assert score == pytest.approx(0.25)

    def test_tool_wear_alone(self) -> None:
        score = defect_risk_index(temperature=50.0, vibration=1.0, tool_wear=50.0, pressure=30.0)
        assert score == pytest.approx(0.25)

    def test_pressure_alone(self) -> None:
        score = defect_risk_index(temperature=50.0, vibration=1.0, tool_wear=10.0, pressure=60.0)
        assert score == pytest.approx(0.15)

    def test_in_unit_interval(self) -> None:
        score = defect_risk_index(temperature=70.0, vibration=4.0, tool_wear=35.0, pressure=50.0)
        assert 0.0 <= score <= 1.0


class TestSensorRangeFeature:
    def test_returns_dict(self) -> None:
        row = {"temperature": 70.0, "vibration": 3.0, "pressure": 45.0, "tool_wear": 25.0}
        result = sensor_range_feature(row)
        assert isinstance(result, dict)

    def test_has_range_key(self) -> None:
        row = {"temperature": 70.0, "vibration": 3.0, "pressure": 45.0, "tool_wear": 25.0}
        result = sensor_range_feature(row)
        assert any("range" in k for k in result)

    def test_range_non_negative(self) -> None:
        row = {"temperature": 70.0, "vibration": 3.0, "pressure": 45.0, "tool_wear": 25.0}
        result = sensor_range_feature(row)
        for v in result.values():
            assert v >= 0.0

    def test_empty_row_returns_dict(self) -> None:
        result = sensor_range_feature({})
        assert isinstance(result, dict)
