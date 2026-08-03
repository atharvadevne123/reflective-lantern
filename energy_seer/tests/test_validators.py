"""Tests for energy_seer/app/validators.py."""
from __future__ import annotations

import pytest


class TestClamp:
    def test_clamps_below(self) -> None:
        from energy_seer.app.validators import clamp
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_clamps_above(self) -> None:
        from energy_seer.app.validators import clamp
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_within_range(self) -> None:
        from energy_seer.app.validators import clamp
        assert clamp(5.0, 0.0, 10.0) == 5.0


class TestValidateMeterID:
    def test_valid(self) -> None:
        from energy_seer.app.validators import validate_meter_id
        assert validate_meter_id("  METER-001  ") == "METER-001"

    def test_empty_raises(self) -> None:
        from energy_seer.app.validators import validate_meter_id
        with pytest.raises(ValueError):
            validate_meter_id("")

    def test_too_long_raises(self) -> None:
        from energy_seer.app.validators import validate_meter_id
        with pytest.raises(ValueError):
            validate_meter_id("x" * 65)


class TestValidateForecastLength:
    def test_valid(self) -> None:
        from energy_seer.app.validators import validate_forecast_length
        assert validate_forecast_length(24) == 24

    def test_zero_raises(self) -> None:
        from energy_seer.app.validators import validate_forecast_length
        with pytest.raises(ValueError):
            validate_forecast_length(0)

    def test_exceeds_max_raises(self) -> None:
        from energy_seer.app.validators import validate_forecast_length
        with pytest.raises(ValueError):
            validate_forecast_length(9000)

    def test_custom_max(self) -> None:
        from energy_seer.app.validators import validate_forecast_length
        with pytest.raises(ValueError):
            validate_forecast_length(100, max_length=50)


class TestValidateTariff:
    def test_valid(self) -> None:
        from energy_seer.app.validators import validate_tariff
        assert validate_tariff(0.15) == pytest.approx(0.15)

    def test_zero_ok(self) -> None:
        from energy_seer.app.validators import validate_tariff
        assert validate_tariff(0.0) == 0.0

    def test_negative_raises(self) -> None:
        from energy_seer.app.validators import validate_tariff
        with pytest.raises(ValueError):
            validate_tariff(-0.01)

    def test_nan_raises(self) -> None:
        from energy_seer.app.validators import validate_tariff
        with pytest.raises(ValueError):
            validate_tariff(float("nan"))


class TestValidateReadingsList:
    def test_valid(self) -> None:
        from energy_seer.app.validators import validate_readings_list
        result = validate_readings_list([1.0, 2.0, 3.0])
        assert result == [1.0, 2.0, 3.0]

    def test_empty_raises(self) -> None:
        from energy_seer.app.validators import validate_readings_list
        with pytest.raises(ValueError):
            validate_readings_list([])

    def test_negative_raises_by_default(self) -> None:
        from energy_seer.app.validators import validate_readings_list
        with pytest.raises(ValueError):
            validate_readings_list([-1.0])

    def test_negative_allowed(self) -> None:
        from energy_seer.app.validators import validate_readings_list
        result = validate_readings_list([-1.0, 2.0], allow_negative=True)
        assert result[0] == -1.0

    def test_nan_raises(self) -> None:
        from energy_seer.app.validators import validate_readings_list
        with pytest.raises(ValueError):
            validate_readings_list([1.0, float("nan")])
