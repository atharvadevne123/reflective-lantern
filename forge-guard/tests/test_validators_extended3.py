"""Extended sensor validator tests — third pass with boundary and parametrize."""

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


def test_valid_reading_no_errors() -> None:
    assert validate_sensor_reading(_valid_reading()) == []


def test_missing_field_reported() -> None:
    data = _valid_reading()
    del data["temperature"]
    errors = validate_sensor_reading(data)
    assert any("temperature" in e for e in errors)


@pytest.mark.parametrize("field", list(SENSOR_RANGES.keys()))
def test_each_field_missing_reports_error(field: str) -> None:
    data = _valid_reading()
    del data[field]
    errors = validate_sensor_reading(data)
    assert any(field in e for e in errors)


def test_temperature_above_max_is_error() -> None:
    data = _valid_reading()
    data["temperature"] = 9999.0
    assert validate_sensor_reading(data)


def test_is_valid_true_for_good_reading() -> None:
    assert is_valid_sensor_reading(_valid_reading())


def test_is_valid_false_for_bad_reading() -> None:
    data = _valid_reading()
    data["pressure"] = -1.0
    assert not is_valid_sensor_reading(data)


def test_sanitize_fills_missing_field() -> None:
    data = {k: v for k, v in _valid_reading().items() if k != "humidity"}
    result = sanitize_sensor_reading(data)
    assert "humidity" in result
    assert math.isfinite(result["humidity"])


def test_sanitize_clamps_above_max() -> None:
    data = _valid_reading()
    data["temperature"] = 9999.0
    result = sanitize_sensor_reading(data)
    _, hi = SENSOR_RANGES["temperature"]
    assert result["temperature"] <= hi


def test_sensor_z_score_positive_above_mean() -> None:
    assert sensor_z_score(10.0, 5.0, 2.0) == pytest.approx(2.5)


def test_sensor_z_score_zero_std_raises() -> None:
    with pytest.raises(ValueError):
        sensor_z_score(10.0, 5.0, 0.0)


def test_clamp_to_range_valid_field() -> None:
    result = clamp_to_range("temperature", 500.0)
    _, hi = SENSOR_RANGES["temperature"]
    assert result == hi


def test_clamp_to_range_unknown_field_raises() -> None:
    with pytest.raises(KeyError):
        clamp_to_range("unknown_field", 1.0)


def test_sensor_drift_detected_large_shift() -> None:
    baseline = [50.0] * 10
    current = [100.0] * 10
    assert sensor_drift_detected(baseline, current, threshold=0.1)


def test_sensor_drift_no_drift_same_data() -> None:
    data = [50.0] * 20
    assert not sensor_drift_detected(data, data, threshold=0.1)


def test_sensor_drift_empty_raises() -> None:
    with pytest.raises(ValueError):
        sensor_drift_detected([], [1.0])


def test_zscore_outlier_extreme_value() -> None:
    values = [0.0] * 99 + [1000.0]
    mask = zscore_outlier(values)
    assert mask[-1] is True


def test_zscore_outlier_uniform_all_false() -> None:
    values = [5.0] * 20
    assert all(v is False for v in zscore_outlier(values))


def test_zscore_outlier_empty_raises() -> None:
    with pytest.raises(ValueError):
        zscore_outlier([])


def test_missing_rate_zero_when_all_present() -> None:
    records = [{"temperature": 75.0}] * 5
    assert missing_rate(records, "temperature") == pytest.approx(0.0)


def test_missing_rate_one_when_all_missing() -> None:
    records = [{} for _ in range(5)]
    assert missing_rate(records, "temperature") == pytest.approx(1.0)


def test_missing_rate_empty_records_zero() -> None:
    assert missing_rate([], "temperature") == pytest.approx(0.0)


def test_clamp_reading_clips_low() -> None:
    assert clamp_reading(-5.0, 0.0, 100.0) == pytest.approx(0.0)


def test_clamp_reading_clips_high() -> None:
    assert clamp_reading(200.0, 0.0, 100.0) == pytest.approx(100.0)


def test_clamp_reading_lo_greater_than_hi_raises() -> None:
    with pytest.raises(ValueError):
        clamp_reading(50.0, 100.0, 0.0)


def test_validate_batch_all_valid() -> None:
    result = validate_batch([_valid_reading()] * 5)
    assert result["valid_count"] == 5
    assert result["invalid_count"] == 0


def test_validate_batch_empty() -> None:
    result = validate_batch([])
    assert result["total"] == 0


def test_field_statistics_basic() -> None:
    records = [{"temperature": float(i)} for i in range(5)]
    stats = field_statistics(records, "temperature")
    assert "mean" in stats
    assert "std" in stats


def test_field_statistics_empty_records_zeros() -> None:
    stats = field_statistics([], "temperature")
    assert stats["mean"] == pytest.approx(0.0)
