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


class TestSensorZScore:
    def test_zero_mean_positive_value(self):
        from app.validators import sensor_z_score

        assert sensor_z_score(2.0, mean=0.0, std=1.0) == pytest.approx(2.0)

    def test_negative_z_score(self):
        from app.validators import sensor_z_score

        assert sensor_z_score(70.0, mean=75.0, std=5.0) == pytest.approx(-1.0)

    def test_zero_std_raises(self):
        from app.validators import sensor_z_score

        with pytest.raises(ValueError, match="std"):
            sensor_z_score(1.0, mean=0.0, std=0.0)

    @pytest.mark.parametrize(
        "value,mean,std,expected",
        [
            (10.0, 10.0, 1.0, 0.0),
            (12.0, 10.0, 2.0, 1.0),
            (8.0, 10.0, 2.0, -1.0),
        ],
    )
    def test_parametrized(self, value, mean, std, expected):
        from app.validators import sensor_z_score

        assert sensor_z_score(value, mean, std) == pytest.approx(expected)


class TestClampToRange:
    def test_value_within_range_unchanged(self):
        from app.validators import clamp_to_range

        assert clamp_to_range("temperature", 75.0) == pytest.approx(75.0)

    def test_value_below_min_clamped(self):
        from app.validators import clamp_to_range

        result = clamp_to_range("temperature", -100.0)
        assert result >= 0.0

    def test_value_above_max_clamped(self):
        from app.validators import clamp_to_range

        result = clamp_to_range("temperature", 9999.0)
        assert result <= 200.0

    def test_unknown_field_raises(self):
        from app.validators import clamp_to_range

        with pytest.raises(KeyError):
            clamp_to_range("unknown_sensor", 1.0)


class TestSensorDriftDetected:
    def test_no_drift_returns_false(self):
        from app.validators import sensor_drift_detected

        baseline = [10.0] * 20
        current = [10.0] * 20
        assert sensor_drift_detected(baseline, current) is False

    def test_large_drift_returns_true(self):
        from app.validators import sensor_drift_detected

        baseline = [10.0] * 20
        current = [20.0] * 20
        assert sensor_drift_detected(baseline, current) is True

    def test_empty_baseline_raises(self):
        from app.validators import sensor_drift_detected

        with pytest.raises(ValueError):
            sensor_drift_detected([], [1.0, 2.0])

    def test_empty_current_raises(self):
        from app.validators import sensor_drift_detected

        with pytest.raises(ValueError):
            sensor_drift_detected([1.0, 2.0], [])

    @pytest.mark.parametrize(
        "baseline,current,threshold,expected",
        [
            ([10.0] * 10, [10.5] * 10, 0.15, False),
            ([10.0] * 10, [15.0] * 10, 0.10, True),
        ],
    )
    def test_parametrized_drift(self, baseline, current, threshold, expected):
        from app.validators import sensor_drift_detected

        assert sensor_drift_detected(baseline, current, threshold=threshold) == expected


class TestZscoreOutlier:
    def test_no_outliers(self) -> None:
        from app.validators import zscore_outlier

        result = zscore_outlier([1.0, 2.0, 3.0, 4.0, 5.0], threshold=3.0)
        assert not any(result)

    def test_detects_outlier(self) -> None:
        from app.validators import zscore_outlier

        result = zscore_outlier([1.0, 1.0, 1.0, 1.0, 100.0], threshold=2.0)
        assert result[-1] is True

    def test_constant_series_no_outliers(self) -> None:
        from app.validators import zscore_outlier

        assert zscore_outlier([5.0, 5.0, 5.0]) == [False, False, False]

    def test_empty_raises(self) -> None:
        from app.validators import zscore_outlier

        with pytest.raises(ValueError):
            zscore_outlier([])


class TestMissingRate:
    def test_no_missing(self) -> None:
        from app.validators import missing_rate

        records = [{"a": 1}, {"a": 2}]
        assert missing_rate(records, "a") == 0.0

    def test_all_missing(self) -> None:
        from app.validators import missing_rate

        records = [{"a": None}, {"b": 1}]
        assert missing_rate(records, "a") == pytest.approx(1.0, abs=0.01)

    def test_half_missing(self) -> None:
        from app.validators import missing_rate

        records = [{"x": 1}, {"x": None}]
        assert missing_rate(records, "x") == pytest.approx(0.5, abs=0.01)

    def test_empty_records_returns_zero(self) -> None:
        from app.validators import missing_rate

        assert missing_rate([], "x") == 0.0


class TestClampReading:
    def test_within_range(self) -> None:
        from app.validators import clamp_reading

        assert clamp_reading(5.0, 0.0, 10.0) == 5.0

    def test_below_lo(self) -> None:
        from app.validators import clamp_reading

        assert clamp_reading(-5.0, 0.0, 10.0) == 0.0

    def test_above_hi(self) -> None:
        from app.validators import clamp_reading

        assert clamp_reading(15.0, 0.0, 10.0) == 10.0

    def test_lo_gt_hi_raises(self) -> None:
        from app.validators import clamp_reading

        with pytest.raises(ValueError):
            clamp_reading(5.0, 10.0, 0.0)
