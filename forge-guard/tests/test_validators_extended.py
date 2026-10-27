"""Extended tests for forge-guard app/validators.py."""

from __future__ import annotations

import math

import pytest

from app.validators import (
    SENSOR_RANGES,
    clamp_reading,
    clamp_to_range,
    field_statistics,
    is_valid_sensor_reading,
    missing_rate,
    sanitize_sensor_reading,
    sensor_drift_detected,
    sensor_z_score,
    validate_batch,
    validate_sensor_reading,
    zscore_outlier,
)


def _valid_reading() -> dict:
    return {
        "temperature": 75.0,
        "pressure": 50.0,
        "vibration": 3.0,
        "cycle_time": 30.0,
        "tool_wear": 20.0,
        "power_consumption": 100.0,
        "humidity": 50.0,
    }


class TestValidateSensorReading:
    def test_valid_reading_no_errors(self) -> None:
        assert validate_sensor_reading(_valid_reading()) == []

    def test_missing_field_reported(self) -> None:
        data = _valid_reading()
        del data["temperature"]
        errors = validate_sensor_reading(data)
        assert any("temperature" in e for e in errors)

    def test_nan_value_reported(self) -> None:
        data = _valid_reading()
        data["vibration"] = math.nan
        errors = validate_sensor_reading(data)
        assert any("vibration" in e for e in errors)

    def test_out_of_range_reported(self) -> None:
        data = _valid_reading()
        data["temperature"] = 999.0
        errors = validate_sensor_reading(data)
        assert any("temperature" in e for e in errors)

    def test_implausible_cross_field_detected(self) -> None:
        data = _valid_reading()
        data["vibration"] = 45.0
        data["cycle_time"] = 0.5
        errors = validate_sensor_reading(data)
        assert any("Implausible" in e for e in errors)


class TestIsValidSensorReading:
    def test_valid_returns_true(self) -> None:
        assert is_valid_sensor_reading(_valid_reading()) is True

    def test_invalid_returns_false(self) -> None:
        data = _valid_reading()
        data["temperature"] = 9999.0
        assert is_valid_sensor_reading(data) is False


class TestSensorZScore:
    def test_mean_value_gives_zero(self) -> None:
        assert sensor_z_score(5.0, 5.0, 1.0) == pytest.approx(0.0)

    def test_one_std_above_gives_one(self) -> None:
        assert sensor_z_score(6.0, 5.0, 1.0) == pytest.approx(1.0)

    def test_zero_std_raises(self) -> None:
        with pytest.raises(ValueError):
            sensor_z_score(5.0, 5.0, 0.0)


class TestClampToRange:
    def test_value_in_range_unchanged(self) -> None:
        assert clamp_to_range("temperature", 100.0) == pytest.approx(100.0)

    def test_value_below_lo_clamped(self) -> None:
        lo, _ = SENSOR_RANGES["temperature"]
        assert clamp_to_range("temperature", lo - 100.0) == pytest.approx(lo)

    def test_value_above_hi_clamped(self) -> None:
        _, hi = SENSOR_RANGES["temperature"]
        assert clamp_to_range("temperature", hi + 100.0) == pytest.approx(hi)

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(KeyError):
            clamp_to_range("unknown_sensor", 50.0)


class TestSensorDriftDetected:
    def test_no_drift_similar_data(self) -> None:
        baseline = [100.0] * 20
        current = [101.0] * 20
        assert sensor_drift_detected(baseline, current) is False

    def test_drift_detected_large_shift(self) -> None:
        baseline = [10.0] * 20
        current = [50.0] * 20
        assert sensor_drift_detected(baseline, current) is True

    def test_empty_baseline_raises(self) -> None:
        with pytest.raises(ValueError):
            sensor_drift_detected([], [1.0, 2.0])


class TestZscoreOutlier:
    def test_all_false_for_uniform(self) -> None:
        result = zscore_outlier([5.0] * 10)
        assert all(v is False for v in result)

    def test_extreme_outlier_detected(self) -> None:
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]
        result = zscore_outlier(values)
        assert result[-1] is True

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            zscore_outlier([])


class TestMissingRate:
    def test_no_missing(self) -> None:
        records = [{"x": 1.0}, {"x": 2.0}]
        assert missing_rate(records, "x") == pytest.approx(0.0)

    def test_all_missing(self) -> None:
        records = [{"y": 1.0}, {"y": 2.0}]
        assert missing_rate(records, "x") == pytest.approx(1.0)

    def test_empty_records_zero(self) -> None:
        assert missing_rate([], "x") == pytest.approx(0.0)


class TestValidateBatch:
    def test_all_valid_no_errors(self) -> None:
        records = [_valid_reading(), _valid_reading()]
        result = validate_batch(records)
        assert result["valid_count"] == 2
        assert result["invalid_count"] == 0

    def test_invalid_record_counted(self) -> None:
        bad = _valid_reading()
        bad["temperature"] = 9999.0
        result = validate_batch([_valid_reading(), bad])
        assert result["invalid_count"] == 1

    def test_total_matches_input(self) -> None:
        records = [_valid_reading()] * 5
        result = validate_batch(records)
        assert result["total"] == 5


class TestFieldStatistics:
    def test_stats_for_constant_field(self) -> None:
        records = [{"temp": 10.0}] * 5
        stats = field_statistics(records, "temp")
        assert stats["mean"] == pytest.approx(10.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_empty_or_missing_field_returns_zeros(self) -> None:
        stats = field_statistics([], "temp")
        assert stats["mean"] == pytest.approx(0.0)


class TestClampReading:
    def test_clamped_below(self) -> None:
        assert clamp_reading(-5.0, 0.0, 100.0) == pytest.approx(0.0)

    def test_clamped_above(self) -> None:
        assert clamp_reading(200.0, 0.0, 100.0) == pytest.approx(100.0)

    def test_in_range_unchanged(self) -> None:
        assert clamp_reading(50.0, 0.0, 100.0) == pytest.approx(50.0)

    def test_lo_gt_hi_raises(self) -> None:
        with pytest.raises(ValueError):
            clamp_reading(5.0, 10.0, 1.0)
