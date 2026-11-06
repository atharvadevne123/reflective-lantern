"""Extended tests for energy_seer app/validators.py — additional edge cases."""

from __future__ import annotations

import pytest

from app.validators import (
    clamp,
    validate_alpha,
    validate_feature_window,
    validate_horizon,
    validate_humidity,
    validate_meter_id,
    validate_period,
    validate_readings_list,
    validate_tariff,
    validate_temperature,
)


class TestClampExtended:
    def test_lo_equals_hi_returns_lo(self) -> None:
        assert clamp(5.0, 3.0, 3.0) == 3.0

    def test_negative_range(self) -> None:
        assert clamp(-2.0, -10.0, -1.0) == -2.0

    def test_value_at_lo_boundary(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0

    def test_value_at_hi_boundary(self) -> None:
        assert clamp(10.0, 0.0, 10.0) == 10.0

    @pytest.mark.parametrize(
        "v,lo,hi,expected",
        [
            (float("inf"), 0.0, 100.0, 100.0),
            (float("-inf"), 0.0, 100.0, 0.0),
        ],
    )
    def test_infinity_clamped(self, v: float, lo: float, hi: float, expected: float) -> None:
        assert clamp(v, lo, hi) == expected


class TestValidateMeterIdExtended:
    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_meter_id("   ")

    def test_exactly_64_chars_accepted(self) -> None:
        meter_id = "M" * 64
        assert validate_meter_id(meter_id) == meter_id

    def test_65_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="64"):
            validate_meter_id("M" * 65)

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert validate_meter_id("  METER-01  ") == "METER-01"

    @pytest.mark.parametrize("meter_id", ["A", "1", "X" * 64])
    def test_valid_meter_ids(self, meter_id: str) -> None:
        assert validate_meter_id(meter_id) == meter_id.strip()


class TestValidateAlphaExtended:
    def test_alpha_one_accepted(self) -> None:
        assert validate_alpha(1.0) == pytest.approx(1.0)

    def test_alpha_very_small_accepted(self) -> None:
        assert validate_alpha(0.001) == pytest.approx(0.001)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(0.0)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(1.0001)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(-0.5)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_alpha(float("nan"))

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9, 1.0])
    def test_valid_alpha_values(self, alpha: float) -> None:
        assert validate_alpha(alpha) == pytest.approx(alpha)


class TestValidatePeriodExtended:
    def test_period_equals_min_accepted(self) -> None:
        assert validate_period(2, min_period=2) == 2

    def test_period_below_default_min_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_period(1)

    def test_large_period_accepted(self) -> None:
        assert validate_period(8760) == 8760

    def test_custom_min_period_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_period(1, min_period=2)

    @pytest.mark.parametrize("period", [2, 7, 12, 24, 52, 365])
    def test_common_seasonal_periods(self, period: int) -> None:
        assert validate_period(period) == period


class TestValidateTemperatureExtended:
    def test_zero_celsius_accepted(self) -> None:
        assert validate_temperature(0.0) == pytest.approx(0.0)

    def test_exactly_negative_60_accepted(self) -> None:
        assert validate_temperature(-60.0) == pytest.approx(-60.0)

    def test_exactly_60_accepted(self) -> None:
        assert validate_temperature(60.0) == pytest.approx(60.0)

    def test_negative_61_raises(self) -> None:
        with pytest.raises(ValueError, match="-60"):
            validate_temperature(-61.0)

    def test_positive_61_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_temperature(61.0)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_temperature(float("nan"))

    def test_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_temperature(float("inf"))

    @pytest.mark.parametrize("temp", [-60.0, -30.0, 0.0, 20.0, 37.0, 60.0])
    def test_common_temperatures(self, temp: float) -> None:
        assert validate_temperature(temp) == pytest.approx(temp)


class TestValidateHumidityExtended:
    def test_zero_pct_accepted(self) -> None:
        assert validate_humidity(0.0) == pytest.approx(0.0)

    def test_100_pct_accepted(self) -> None:
        assert validate_humidity(100.0) == pytest.approx(100.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_humidity(-0.001)

    def test_above_100_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_humidity(100.001)

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_humidity(float("nan"))

    @pytest.mark.parametrize("h", [0.0, 25.0, 50.0, 75.0, 100.0])
    def test_valid_humidity_values(self, h: float) -> None:
        assert validate_humidity(h) == pytest.approx(h)


class TestValidateHorizonExtended:
    def test_one_hour_accepted(self) -> None:
        assert validate_horizon(1) == 1

    def test_168_hours_accepted(self) -> None:
        assert validate_horizon(168) == 168

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="1 hour"):
            validate_horizon(0)

    def test_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="maximum"):
            validate_horizon(169, max_hours=168)

    def test_custom_max_accepted(self) -> None:
        assert validate_horizon(500, max_hours=720) == 500

    @pytest.mark.parametrize("h", [1, 6, 12, 24, 48, 72, 168])
    def test_common_forecast_horizons(self, h: int) -> None:
        assert validate_horizon(h) == h


class TestValidateReadingsListExtended:
    def test_single_value_list_accepted(self) -> None:
        assert validate_readings_list([0.0]) == [0.0]

    def test_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_readings_list([1.0, float("inf")])

    def test_negative_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_readings_list([float("-inf")])

    def test_nan_at_first_index_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_readings_list([float("nan"), 1.0])

    def test_all_zeros_allowed(self) -> None:
        result = validate_readings_list([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]

    def test_large_list_accepted(self) -> None:
        readings = [float(i) for i in range(1000)]
        result = validate_readings_list(readings)
        assert len(result) == 1000

    def test_allow_negative_true_accepts_negatives(self) -> None:
        readings = [-5.0, -1.0, 0.0, 1.0]
        result = validate_readings_list(readings, allow_negative=True)
        assert result[0] == -5.0


class TestValidateTariffExtended:
    def test_zero_tariff_accepted(self) -> None:
        assert validate_tariff(0.0) == 0.0

    def test_very_large_tariff_accepted(self) -> None:
        assert validate_tariff(9999.99) == pytest.approx(9999.99)

    def test_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_tariff(float("inf"))

    def test_negative_infinity_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_tariff(float("-inf"))

    @pytest.mark.parametrize("t", [0.0, 0.01, 0.12, 0.25, 1.0])
    def test_common_tariffs(self, t: float) -> None:
        assert validate_tariff(t) == pytest.approx(t)


class TestValidateFeatureWindowExtended:
    def test_exactly_min_length_accepted(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert validate_feature_window(values, min_length=5) == values

    def test_min_length_1_accepts_single_value(self) -> None:
        assert validate_feature_window([42.0], min_length=1) == [42.0]

    def test_large_window_accepted(self) -> None:
        values = list(range(100))
        result = validate_feature_window(values, min_length=10)
        assert len(result) == 100

    def test_values_returned_unchanged(self) -> None:
        values = [3.14, 2.71, 1.41]
        assert validate_feature_window(values, min_length=3) == values

    @pytest.mark.parametrize("min_len", [1, 3, 5, 10, 24])
    def test_various_min_lengths(self, min_len: int) -> None:
        values = [1.0] * min_len
        assert validate_feature_window(values, min_length=min_len) == values
