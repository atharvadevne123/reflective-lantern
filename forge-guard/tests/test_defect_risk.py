"""Focused tests for the sensor-summary and defect-risk helpers in app/features.py.

These are pure functions over a single reading, so they are pinned here
independently of the sklearn transformer tests in tests/test_features.py.
"""

from __future__ import annotations

import pytest

from app.features import defect_risk_index, sensor_range_feature

# Thresholds the risk index keys off, with their individual weights.
NOMINAL = {"temperature": 70.0, "vibration": 3.0, "tool_wear": 20.0, "pressure": 50.0}
ALL_TRIPPED = {"temperature": 90.0, "vibration": 6.0, "tool_wear": 50.0, "pressure": 60.0}


class TestSensorRangeFeature:
    def test_computes_min_max_and_range(self) -> None:
        result = sensor_range_feature({"a": 1.0, "b": 5.0, "c": 3.0})
        assert result["sensor_min"] == pytest.approx(1.0)
        assert result["sensor_max"] == pytest.approx(5.0)
        assert result["sensor_range"] == pytest.approx(4.0)

    def test_empty_row_returns_zeros(self) -> None:
        assert sensor_range_feature({}) == {
            "sensor_min": 0.0,
            "sensor_max": 0.0,
            "sensor_range": 0.0,
        }

    def test_single_sensor_has_no_range(self) -> None:
        result = sensor_range_feature({"only": 42.0})
        assert result["sensor_min"] == pytest.approx(42.0)
        assert result["sensor_max"] == pytest.approx(42.0)
        assert result["sensor_range"] == 0.0

    def test_identical_readings_have_no_range(self) -> None:
        assert sensor_range_feature({"a": 7.0, "b": 7.0, "c": 7.0})["sensor_range"] == 0.0

    def test_negative_readings_are_handled(self) -> None:
        result = sensor_range_feature({"a": -10.0, "b": 5.0})
        assert result["sensor_min"] == pytest.approx(-10.0)
        assert result["sensor_range"] == pytest.approx(15.0)

    def test_range_is_never_negative(self) -> None:
        for row in ({"a": 1.0}, {"a": -5.0, "b": -1.0}, {"a": 3.0, "b": 3.0}):
            assert sensor_range_feature(row)["sensor_range"] >= 0.0

    def test_range_equals_max_minus_min(self) -> None:
        result = sensor_range_feature({"a": 2.5, "b": 9.75, "c": -1.25})
        assert result["sensor_range"] == pytest.approx(result["sensor_max"] - result["sensor_min"])

    def test_always_returns_the_three_keys(self) -> None:
        for row in ({}, {"a": 1.0}, {"a": 1.0, "b": 2.0}):
            assert set(sensor_range_feature(row)) == {"sensor_min", "sensor_max", "sensor_range"}


class TestDefectRiskIndex:
    def test_nominal_readings_carry_no_risk(self) -> None:
        assert defect_risk_index(**NOMINAL) == 0.0

    def test_all_thresholds_tripped_gives_maximum_risk(self) -> None:
        # The four weights sum to 0.95, so that is the attainable ceiling.
        assert defect_risk_index(**ALL_TRIPPED) == pytest.approx(0.95)

    @pytest.mark.parametrize(
        ("sensor", "tripped_value", "weight"),
        [
            ("temperature", 90.0, 0.30),
            ("vibration", 6.0, 0.25),
            ("tool_wear", 50.0, 0.25),
            ("pressure", 60.0, 0.15),
        ],
    )
    def test_each_sensor_contributes_its_weight(self, sensor: str, tripped_value: float, weight: float) -> None:
        readings = dict(NOMINAL)
        readings[sensor] = tripped_value
        assert defect_risk_index(**readings) == pytest.approx(weight)

    def test_temperature_is_the_heaviest_signal(self) -> None:
        hot = defect_risk_index(**{**NOMINAL, "temperature": 90.0})
        high_pressure = defect_risk_index(**{**NOMINAL, "pressure": 60.0})
        assert hot > high_pressure

    def test_contributions_are_additive(self) -> None:
        both = defect_risk_index(**{**NOMINAL, "temperature": 90.0, "vibration": 6.0})
        assert both == pytest.approx(0.55)

    @pytest.mark.parametrize(
        ("sensor", "boundary"),
        [("temperature", 85.0), ("vibration", 5.0), ("tool_wear", 40.0), ("pressure", 58.0)],
    )
    def test_value_at_threshold_does_not_trip(self, sensor: str, boundary: float) -> None:
        # The comparisons are strictly greater-than, so the boundary is safe.
        assert defect_risk_index(**{**NOMINAL, sensor: boundary}) == 0.0

    @pytest.mark.parametrize(
        ("sensor", "just_over"),
        [("temperature", 85.01), ("vibration", 5.01), ("tool_wear", 40.01), ("pressure", 58.01)],
    )
    def test_value_just_over_threshold_trips(self, sensor: str, just_over: float) -> None:
        assert defect_risk_index(**{**NOMINAL, sensor: just_over}) > 0.0

    def test_index_is_bounded(self) -> None:
        for readings in (NOMINAL, ALL_TRIPPED, {**NOMINAL, "temperature": 1e6}):
            assert 0.0 <= defect_risk_index(**readings) <= 1.0

    def test_extreme_readings_do_not_exceed_the_ceiling(self) -> None:
        # Once every threshold is crossed the score saturates: how far past
        # each threshold a reading sits does not raise it further.
        extreme = {"temperature": 1e6, "vibration": 1e6, "tool_wear": 1e6, "pressure": 1e6}
        assert defect_risk_index(**extreme) == pytest.approx(defect_risk_index(**ALL_TRIPPED))

    def test_risk_never_decreases_as_a_sensor_rises(self) -> None:
        scores = [defect_risk_index(**{**NOMINAL, "temperature": t}) for t in (70.0, 84.0, 86.0, 200.0)]
        assert scores == sorted(scores)
