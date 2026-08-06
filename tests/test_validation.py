"""Tests for input validation utilities."""

from __future__ import annotations

from datetime import datetime

import pytest

from app.validation import (
    extract_temporal_from_datetime,
    is_weekend,
    validate_coordinate,
    validate_load_series,
    validate_price,
    validate_reading_dict,
    validate_temporal_fields,
    validate_weather_fields,
)


class TestValidateTemporalFields:
    def test_valid_inputs_no_errors(self) -> None:
        assert validate_temporal_fields(14, 2, 7) == []

    def test_invalid_hour(self) -> None:
        errors = validate_temporal_fields(25, 0, 1)
        assert any("hour" in e for e in errors)

    def test_invalid_month(self) -> None:
        errors = validate_temporal_fields(12, 0, 13)
        assert any("month" in e for e in errors)

    def test_invalid_dow(self) -> None:
        errors = validate_temporal_fields(12, 7, 6)
        assert any("day_of_week" in e for e in errors)

    @pytest.mark.parametrize("hour", [0, 12, 23])
    def test_boundary_hours_valid(self, hour) -> None:
        assert validate_temporal_fields(hour, 0, 1) == []

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_boundary_months_valid(self, month) -> None:
        assert validate_temporal_fields(10, 0, month) == []


class TestValidateWeatherFields:
    def test_valid_weather_no_errors(self) -> None:
        assert validate_weather_fields(25.0, 60.0) == []

    def test_temperature_too_hot(self) -> None:
        errors = validate_weather_fields(999.0, 60.0)
        assert any("temperature" in e for e in errors)

    def test_humidity_out_of_range(self) -> None:
        errors = validate_weather_fields(20.0, -10.0)
        assert any("humidity" in e for e in errors)

    @pytest.mark.parametrize("temp", [-40.0, 0.0, 60.0])
    def test_boundary_temps_valid(self, temp) -> None:
        assert validate_weather_fields(temp, 50.0) == []


class TestValidateLoadSeries:
    def test_empty_series_no_errors(self) -> None:
        assert validate_load_series([]) == []

    def test_valid_loads_no_errors(self) -> None:
        assert validate_load_series([3000.0, 4000.0, 3500.0]) == []

    def test_out_of_range_load(self) -> None:
        errors = validate_load_series([-100.0, 3000.0])
        assert len(errors) > 0

    def test_nan_detected(self) -> None:
        errors = validate_load_series([float("nan"), 3000.0])
        assert any("NaN" in e for e in errors)


class TestHelpers:
    def test_weekend_saturday(self) -> None:
        assert is_weekend(5) is True

    def test_weekend_sunday(self) -> None:
        assert is_weekend(6) is True

    def test_weekday_monday(self) -> None:
        assert is_weekend(0) is False

    @pytest.mark.parametrize("dow", [0, 1, 2, 3, 4])
    def test_weekdays_not_weekend(self, dow) -> None:
        assert is_weekend(dow) is False

    def test_extract_temporal_from_datetime(self) -> None:
        dt = datetime(2026, 7, 10, 14, 30)
        result = extract_temporal_from_datetime(dt)
        assert result["hour"] == 14
        assert result["month"] == 7
        assert "is_weekend" in result

    def test_extract_temporal_keys_present(self) -> None:
        dt = datetime(2026, 1, 1, 0, 0)
        result = extract_temporal_from_datetime(dt)
        for key in ("hour", "day_of_week", "month", "is_weekend"):
            assert key in result

    @pytest.mark.parametrize(
        "hour,month,dow",
        [
            (0, 1, 0),
            (23, 12, 6),
            (12, 6, 3),
        ],
    )
    def test_validate_temporal_boundary_values(self, hour, month, dow) -> None:
        assert validate_temporal_fields(hour, dow, month) == []


class TestValidateLoadSeriesExtended:
    def test_single_valid_value(self) -> None:
        assert validate_load_series([5000.0]) == []

    def test_all_zeros_is_valid(self) -> None:
        assert validate_load_series([0.0, 0.0, 0.0]) == []

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_inf_detected(self, value) -> None:
        errors = validate_load_series([value])
        assert len(errors) > 0


class TestValidateBuildingId:
    def test_valid_building_id(self) -> None:
        from app.validation import validate_building_id

        assert validate_building_id("bldg-001") == []

    def test_empty_building_id(self) -> None:
        from app.validation import validate_building_id

        errors = validate_building_id("")
        assert len(errors) > 0

    def test_building_id_too_long(self) -> None:
        from app.validation import validate_building_id

        errors = validate_building_id("a" * 65)
        assert len(errors) > 0

    def test_building_id_invalid_chars(self) -> None:
        from app.validation import validate_building_id

        errors = validate_building_id("bldg@123")
        assert len(errors) > 0

    @pytest.mark.parametrize("bid", ["a", "A1", "bldg-001", "BLDG_002"])
    def test_valid_building_ids(self, bid) -> None:
        from app.validation import validate_building_id

        assert validate_building_id(bid) == []


class TestValidateBatchSize:
    def test_valid_batch(self) -> None:
        from app.validation import validate_batch_size

        assert validate_batch_size(50) == []

    def test_oversized_batch(self) -> None:
        from app.validation import validate_batch_size

        errors = validate_batch_size(101)
        assert len(errors) > 0

    def test_zero_batch(self) -> None:
        from app.validation import validate_batch_size

        errors = validate_batch_size(0)
        assert len(errors) > 0

    @pytest.mark.parametrize("n", [1, 50, 100])
    def test_valid_batch_sizes(self, n) -> None:
        from app.validation import validate_batch_size

        assert validate_batch_size(n) == []


class TestValidateConsumptionKwh:
    def test_valid_value(self) -> None:
        from app.validation import validate_consumption_kwh

        assert validate_consumption_kwh(10.5) == []

    def test_negative_value(self) -> None:
        from app.validation import validate_consumption_kwh

        errors = validate_consumption_kwh(-1.0)
        assert len(errors) > 0

    def test_nan_value(self) -> None:
        import math

        from app.validation import validate_consumption_kwh

        errors = validate_consumption_kwh(math.nan)
        assert len(errors) > 0


class TestValidateFeatureVector:
    def test_valid_vector(self) -> None:
        from app.validation import validate_feature_vector

        assert validate_feature_vector([1.0, 2.0, 3.0]) == []

    def test_empty_vector(self) -> None:
        from app.validation import validate_feature_vector

        errors = validate_feature_vector([])
        assert len(errors) > 0

    def test_wrong_dim(self) -> None:
        from app.validation import validate_feature_vector

        errors = validate_feature_vector([1.0, 2.0, 3.0], expected_dim=5)
        assert len(errors) > 0

    def test_correct_dim(self) -> None:
        from app.validation import validate_feature_vector

        assert validate_feature_vector([1.0, 2.0, 3.0], expected_dim=3) == []


class TestValidateBuildingIdEdgeCases:
    def test_max_length_allowed(self) -> None:
        from app.validation import MAX_BUILDING_ID_LEN, validate_building_id

        bid = "x" * MAX_BUILDING_ID_LEN
        assert validate_building_id(bid) == []

    def test_over_max_length_rejected(self) -> None:
        from app.validation import MAX_BUILDING_ID_LEN, validate_building_id

        bid = "x" * (MAX_BUILDING_ID_LEN + 1)
        assert len(validate_building_id(bid)) > 0

    def test_alphanumeric_with_hyphens_valid(self) -> None:
        from app.validation import validate_building_id

        assert validate_building_id("bldg-001-main") == []

    def test_special_chars_rejected(self) -> None:
        from app.validation import validate_building_id

        assert len(validate_building_id("bldg@001!")) > 0


class TestValidateConsumptionEdgeCases:
    def test_exact_zero_valid(self) -> None:
        from app.validation import validate_consumption_kwh

        assert validate_consumption_kwh(0.0) == []

    def test_exact_max_valid(self) -> None:
        from app.validation import MAX_CONSUMPTION_KWH, validate_consumption_kwh

        assert validate_consumption_kwh(MAX_CONSUMPTION_KWH) == []

    def test_above_max_rejected(self) -> None:
        from app.validation import MAX_CONSUMPTION_KWH, validate_consumption_kwh

        assert len(validate_consumption_kwh(MAX_CONSUMPTION_KWH + 1.0)) > 0

    def test_negative_rejected(self) -> None:
        from app.validation import validate_consumption_kwh

        assert len(validate_consumption_kwh(-0.001)) > 0


def test_batch_validate_readings_all_valid() -> None:
    from app.validation import batch_validate_readings

    readings = [
        {
            "hour": 12,
            "day_of_week": 1,
            "month": 6,
            "temperature_c": 22.0,
            "humidity_pct": 60.0,
            "consumption_kwh": 15.0,
        },
        {"hour": 8, "day_of_week": 0, "month": 3, "temperature_c": 5.0, "humidity_pct": 40.0, "consumption_kwh": 10.0},
    ]
    results = batch_validate_readings(readings)
    assert all(r["valid"] for r in results)
    assert len(results) == 2


def test_batch_validate_readings_detects_errors() -> None:
    from app.validation import batch_validate_readings

    readings = [
        {"hour": 25, "day_of_week": 1, "month": 6, "temperature_c": 22.0, "humidity_pct": 60.0, "consumption_kwh": 5.0},
    ]
    results = batch_validate_readings(readings)
    assert not results[0]["valid"]
    assert any("hour" in e for e in results[0]["errors"])


def test_batch_validate_readings_preserves_index() -> None:
    from app.validation import batch_validate_readings

    readings = [
        {"hour": 0, "day_of_week": 0, "month": 1, "temperature_c": 10.0, "humidity_pct": 50.0, "consumption_kwh": 5.0}
    ] * 5
    results = batch_validate_readings(readings)
    assert [r["index"] for r in results] == list(range(5))


def test_batch_validate_readings_empty() -> None:
    from app.validation import batch_validate_readings

    assert batch_validate_readings([]) == []


def test_batch_validate_readings_bad_temperature() -> None:
    from app.validation import batch_validate_readings

    readings = [
        {
            "hour": 12,
            "day_of_week": 2,
            "month": 7,
            "temperature_c": 200.0,
            "humidity_pct": 50.0,
            "consumption_kwh": 10.0,
        }
    ]
    results = batch_validate_readings(readings)
    assert not results[0]["valid"]
    assert any("temperature_c" in e for e in results[0]["errors"])


@pytest.mark.parametrize("bad_hour", [-1, 24, 100])
def test_batch_validate_readings_bad_hours(bad_hour) -> None:
    from app.validation import batch_validate_readings

    readings = [
        {
            "hour": bad_hour,
            "day_of_week": 0,
            "month": 1,
            "temperature_c": 10.0,
            "humidity_pct": 50.0,
            "consumption_kwh": 5.0,
        }
    ]
    results = batch_validate_readings(readings)
    assert not results[0]["valid"]


def test_validate_forecast_horizon_valid() -> None:
    from app.validation import validate_forecast_horizon

    assert validate_forecast_horizon(24) == []
    assert validate_forecast_horizon(1) == []
    assert validate_forecast_horizon(8760) == []


def test_validate_forecast_horizon_too_small() -> None:
    from app.validation import validate_forecast_horizon

    errors = validate_forecast_horizon(0)
    assert len(errors) > 0
    assert any("1" in e for e in errors)


def test_validate_forecast_horizon_too_large() -> None:
    from app.validation import validate_forecast_horizon

    errors = validate_forecast_horizon(9000, max_horizon=8760)
    assert len(errors) > 0


def test_validate_forecast_horizon_custom_max() -> None:
    from app.validation import validate_forecast_horizon

    assert validate_forecast_horizon(48, max_horizon=24) != []
    assert validate_forecast_horizon(24, max_horizon=24) == []


@pytest.mark.parametrize(
    "horizon,expected_valid",
    [
        (1, True),
        (24, True),
        (168, True),
        (8760, True),
        (0, False),
        (-1, False),
        (8761, False),
    ],
)
def test_validate_forecast_horizon_parametrized(horizon, expected_valid) -> None:
    from app.validation import validate_forecast_horizon

    errors = validate_forecast_horizon(horizon)
    assert (len(errors) == 0) == expected_valid


def test_is_valid_temporal_input_valid() -> None:
    from app.validation import is_valid_temporal_input

    assert is_valid_temporal_input(12, 3, 6) is True


def test_is_valid_temporal_input_bad_hour() -> None:
    from app.validation import is_valid_temporal_input

    assert is_valid_temporal_input(25, 3, 6) is False


def test_is_valid_temporal_input_bad_month() -> None:
    from app.validation import is_valid_temporal_input

    assert is_valid_temporal_input(0, 0, 13) is False


def test_is_valid_temporal_input_bad_dow() -> None:
    from app.validation import is_valid_temporal_input

    assert is_valid_temporal_input(0, 7, 1) is False


def test_clamp_consumption_within_range() -> None:
    from app.validation import clamp_consumption

    assert clamp_consumption(500.0) == pytest.approx(500.0)


def test_clamp_consumption_below_min() -> None:
    from app.validation import clamp_consumption

    assert clamp_consumption(-10.0) == pytest.approx(0.0)


def test_clamp_consumption_above_max() -> None:
    from app.validation import clamp_consumption

    assert clamp_consumption(999_999.0) == pytest.approx(100_000.0)


@pytest.mark.parametrize(
    "hour,dow,month,expected",
    [
        (0, 0, 1, True),
        (23, 6, 12, True),
        (24, 0, 1, False),
        (0, 0, 0, False),
    ],
)
def test_is_valid_temporal_input_parametrized(hour, dow, month, expected) -> None:
    from app.validation import is_valid_temporal_input

    assert is_valid_temporal_input(hour, dow, month) is expected


def test_clamp_consumption_below_min_v2() -> None:
    from app.validation import clamp_consumption
    result = clamp_consumption(-10.0)
    assert result == 0.0


def test_clamp_consumption_above_max_v2() -> None:
    from app.validation import MAX_CONSUMPTION_KWH, clamp_consumption
    result = clamp_consumption(MAX_CONSUMPTION_KWH + 1000)
    assert result == MAX_CONSUMPTION_KWH


def test_clamp_consumption_in_range_v2() -> None:
    from app.validation import clamp_consumption
    result = clamp_consumption(500.0)
    assert result == 500.0


def test_is_valid_temporal_input_valid_v2() -> None:
    from app.validation import is_valid_temporal_input
    assert is_valid_temporal_input(12, 3, 6) is True


def test_is_valid_temporal_input_invalid_hour() -> None:
    from app.validation import is_valid_temporal_input
    assert is_valid_temporal_input(25, 3, 6) is False


def test_extract_temporal_from_datetime() -> None:
    from datetime import datetime

    from app.validation import extract_temporal_from_datetime
    dt = datetime(2026, 6, 15, 14, 30)
    result = extract_temporal_from_datetime(dt)
    assert result["hour"] == 14
    assert result["month"] == 6
    assert "is_weekend" in result


@pytest.mark.parametrize("horizon,expected_valid", [
    (1, True),
    (24, True),
    (8760, True),
    (0, False),
    (8761, False),
])
def test_validate_forecast_horizon_parametrized_v2(horizon, expected_valid) -> None:
    from app.validation import validate_forecast_horizon
    errors = validate_forecast_horizon(horizon)
    assert (len(errors) == 0) == expected_valid


def test_batch_validate_readings_mixed() -> None:
    from app.validation import batch_validate_readings
    readings = [
        {"hour": 10, "day_of_week": 1, "month": 6, "temperature_c": 22.0, "humidity_pct": 55.0, "consumption_kwh": 12.5},
        {"hour": 99, "day_of_week": 1, "month": 6, "temperature_c": 22.0, "humidity_pct": 55.0, "consumption_kwh": 12.5},
    ]
    results = batch_validate_readings(readings)
    assert results[0]["valid"] is True
    assert results[1]["valid"] is False


class TestValidateReadingDict:
    VALID_READING = {
        "hour": 10,
        "day_of_week": 2,
        "month": 6,
        "temperature_c": 22.0,
        "humidity_pct": 55.0,
        "consumption_kwh": 15.0,
        "building_id": "bldg-001",
    }

    def test_valid_reading_passes(self) -> None:
        result = validate_reading_dict(self.VALID_READING)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_consumption_fails(self) -> None:
        reading = {k: v for k, v in self.VALID_READING.items() if k != "consumption_kwh"}
        result = validate_reading_dict(reading)
        assert result["valid"] is False
        assert any("consumption_kwh" in e for e in result["errors"])

    def test_invalid_hour_fails(self) -> None:
        result = validate_reading_dict({**self.VALID_READING, "hour": 99})
        assert result["valid"] is False

    def test_invalid_temperature_fails(self) -> None:
        result = validate_reading_dict({**self.VALID_READING, "temperature_c": 200.0})
        assert result["valid"] is False

    def test_result_has_required_keys(self) -> None:
        result = validate_reading_dict(self.VALID_READING)
        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result

    @pytest.mark.parametrize("field,value", [
        ("hour", -1),
        ("hour", 24),
        ("month", 0),
        ("month", 13),
        ("day_of_week", 7),
    ])
    def test_out_of_range_temporal_fails(self, field: str, value: int) -> None:
        result = validate_reading_dict({**self.VALID_READING, field: value})
        assert result["valid"] is False


class TestValidateCoordinate:
    def test_valid_coord(self) -> None:
        assert validate_coordinate(37.7749, -122.4194) == []

    def test_invalid_lat_too_high(self) -> None:
        errors = validate_coordinate(91.0, 0.0)
        assert any("latitude" in e for e in errors)

    def test_invalid_lat_too_low(self) -> None:
        errors = validate_coordinate(-91.0, 0.0)
        assert any("latitude" in e for e in errors)

    def test_invalid_lon_too_high(self) -> None:
        errors = validate_coordinate(0.0, 181.0)
        assert any("longitude" in e for e in errors)

    def test_invalid_lon_too_low(self) -> None:
        errors = validate_coordinate(0.0, -181.0)
        assert any("longitude" in e for e in errors)

    def test_boundary_values_valid(self) -> None:
        assert validate_coordinate(90.0, 180.0) == []
        assert validate_coordinate(-90.0, -180.0) == []

    @pytest.mark.parametrize("lat,lon", [
        (0.0, 0.0),
        (51.5074, -0.1278),
        (-33.8688, 151.2093),
    ])
    def test_valid_coords_parametrized(self, lat: float, lon: float) -> None:
        assert validate_coordinate(lat, lon) == []


class TestValidatePrice:
    def test_valid_price(self) -> None:
        assert validate_price(250000.0) == []

    def test_zero_price_valid(self) -> None:
        assert validate_price(0.0) == []

    def test_negative_price_invalid(self) -> None:
        errors = validate_price(-1.0)
        assert any("non-negative" in e for e in errors)

    def test_inf_price_invalid(self) -> None:
        errors = validate_price(float("inf"))
        assert len(errors) > 0

    def test_nan_price_invalid(self) -> None:
        import math
        errors = validate_price(math.nan)
        assert len(errors) > 0

    @pytest.mark.parametrize("price", [0.0, 1.0, 100000.0, 999999999.0])
    def test_valid_prices_parametrized(self, price: float) -> None:
        assert validate_price(price) == []


# Tests for validate_percentage and validate_region_id
import pytest

from app.validation import validate_percentage, validate_region_id


class TestValidatePercentage:
    def test_valid_zero(self) -> None:
        assert validate_percentage(0.0) == []

    def test_valid_100(self) -> None:
        assert validate_percentage(100.0) == []

    def test_valid_midpoint(self) -> None:
        assert validate_percentage(50.0) == []

    def test_below_zero(self) -> None:
        errors = validate_percentage(-1.0)
        assert len(errors) == 1

    def test_above_100(self) -> None:
        errors = validate_percentage(101.0)
        assert len(errors) == 1

    def test_field_name_in_error(self) -> None:
        errors = validate_percentage(-5.0, field_name="savings_pct")
        assert any("savings_pct" in e for e in errors)

    @pytest.mark.parametrize("pct", [0.0, 25.0, 50.0, 75.0, 100.0])
    def test_valid_percentages(self, pct: float) -> None:
        assert validate_percentage(pct) == []


class TestValidateRegionId:
    def test_valid_id(self) -> None:
        assert validate_region_id("northeast") == []

    def test_valid_with_underscore(self) -> None:
        assert validate_region_id("pacific_nw") == []

    def test_empty_id(self) -> None:
        errors = validate_region_id("")
        assert len(errors) > 0

    def test_uppercase_rejected(self) -> None:
        errors = validate_region_id("Northeast")
        assert len(errors) > 0

    def test_special_chars_rejected(self) -> None:
        errors = validate_region_id("north-east")
        assert len(errors) > 0

    @pytest.mark.parametrize("rid", ["northeast", "midwest", "south", "west", "default"])
    def test_valid_known_ids(self, rid: str) -> None:
        assert validate_region_id(rid) == []
