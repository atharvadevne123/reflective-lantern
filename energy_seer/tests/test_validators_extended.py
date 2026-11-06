"""Extended tests for energy_seer app/validators.py — additional edge cases."""

from __future__ import annotations

import pytest

from app.validators import (
    clamp,
    validate_alpha,
    validate_horizon,
    validate_humidity,
    validate_meter_id,
    validate_period,
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
