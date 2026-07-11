"""Tests for input validation utilities."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.validation import (
    extract_temporal_from_datetime,
    is_weekend,
    validate_load_series,
    validate_temporal_fields,
    validate_weather_fields,
)


class TestValidateTemporalFields:
    def test_valid_inputs_no_errors(self):
        assert validate_temporal_fields(14, 2, 7) == []

    def test_invalid_hour(self):
        errors = validate_temporal_fields(25, 0, 1)
        assert any("hour" in e for e in errors)

    def test_invalid_month(self):
        errors = validate_temporal_fields(12, 0, 13)
        assert any("month" in e for e in errors)

    def test_invalid_dow(self):
        errors = validate_temporal_fields(12, 7, 6)
        assert any("day_of_week" in e for e in errors)

    @pytest.mark.parametrize("hour", [0, 12, 23])
    def test_boundary_hours_valid(self, hour):
        assert validate_temporal_fields(hour, 0, 1) == []

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_boundary_months_valid(self, month):
        assert validate_temporal_fields(10, 0, month) == []


class TestValidateWeatherFields:
    def test_valid_weather_no_errors(self):
        assert validate_weather_fields(25.0, 60.0) == []

    def test_temperature_too_hot(self):
        errors = validate_weather_fields(999.0, 60.0)
        assert any("temperature" in e for e in errors)

    def test_humidity_out_of_range(self):
        errors = validate_weather_fields(20.0, -10.0)
        assert any("humidity" in e for e in errors)

    @pytest.mark.parametrize("temp", [-40.0, 0.0, 60.0])
    def test_boundary_temps_valid(self, temp):
        assert validate_weather_fields(temp, 50.0) == []


class TestValidateLoadSeries:
    def test_empty_series_no_errors(self):
        assert validate_load_series([]) == []

    def test_valid_loads_no_errors(self):
        assert validate_load_series([3000.0, 4000.0, 3500.0]) == []

    def test_out_of_range_load(self):
        errors = validate_load_series([-100.0, 3000.0])
        assert len(errors) > 0

    def test_nan_detected(self):
        errors = validate_load_series([float("nan"), 3000.0])
        assert any("NaN" in e for e in errors)


class TestHelpers:
    def test_weekend_saturday(self):
        assert is_weekend(5) is True

    def test_weekend_sunday(self):
        assert is_weekend(6) is True

    def test_weekday_monday(self):
        assert is_weekend(0) is False

    @pytest.mark.parametrize("dow", [0, 1, 2, 3, 4])
    def test_weekdays_not_weekend(self, dow):
        assert is_weekend(dow) is False

    def test_extract_temporal_from_datetime(self):
        dt = datetime(2026, 7, 10, 14, 30)
        result = extract_temporal_from_datetime(dt)
        assert result["hour"] == 14
        assert result["month"] == 7
        assert "is_weekend" in result

    def test_extract_temporal_keys_present(self):
        dt = datetime(2026, 1, 1, 0, 0)
        result = extract_temporal_from_datetime(dt)
        for key in ("hour", "day_of_week", "month", "is_weekend"):
            assert key in result

    @pytest.mark.parametrize("hour,month,dow", [
        (0, 1, 0),
        (23, 12, 6),
        (12, 6, 3),
    ])
    def test_validate_temporal_boundary_values(self, hour, month, dow):
        assert validate_temporal_fields(hour, dow, month) == []


class TestValidateLoadSeriesExtended:
    def test_single_valid_value(self):
        assert validate_load_series([5000.0]) == []

    def test_all_zeros_is_valid(self):
        assert validate_load_series([0.0, 0.0, 0.0]) == []

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_inf_detected(self, value):
        errors = validate_load_series([value])
        assert len(errors) > 0
