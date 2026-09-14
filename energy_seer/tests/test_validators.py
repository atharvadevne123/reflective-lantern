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


@pytest.mark.parametrize("n", [5, 10, 100, 8760])
def test_validate_forecast_length_valid(n: int) -> None:
    from energy_seer.app.validators import validate_forecast_length

    assert validate_forecast_length(n) == n


@pytest.mark.parametrize("n", [0, -1, 8761])
def test_validate_forecast_length_invalid_raises(n: int) -> None:
    import pytest
    from energy_seer.app.validators import validate_forecast_length

    with pytest.raises(ValueError):
        validate_forecast_length(n)


@pytest.mark.parametrize("tariff", [0.01, 0.10, 0.50, 1.00])
def test_validate_tariff_positive_returns_value(tariff: float) -> None:
    from energy_seer.app.validators import validate_tariff

    assert validate_tariff(tariff) == pytest.approx(tariff)


@pytest.mark.parametrize("tariff", [0.0, -0.1, -10.0])
def test_validate_tariff_non_positive_raises(tariff: float) -> None:
    import pytest
    from energy_seer.app.validators import validate_tariff

    with pytest.raises(ValueError):
        validate_tariff(tariff)


@pytest.mark.parametrize(
    "meter_id",
    ["MTR-001", "MTR-999", "BLDG-42-A"],
)
def test_validate_meter_id_valid_returns_string(meter_id: str) -> None:
    from energy_seer.app.validators import validate_meter_id

    result = validate_meter_id(meter_id)
    assert isinstance(result, str)
    assert result == meter_id


@pytest.mark.parametrize(
    "value,lo,hi,expected",
    [
        (5.0, 0.0, 10.0, 5.0),
        (-5.0, 0.0, 10.0, 0.0),
        (15.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 1.0),
    ],
)
def test_clamp_parametrized(value: float, lo: float, hi: float, expected: float) -> None:
    """clamp returns the correct boundary-clamped value."""
    from energy_seer.app.validators import clamp

    assert clamp(value, lo, hi) == expected


@pytest.mark.parametrize("tariff", [0.01, 0.5, 1.0, 100.0])
def test_validate_tariff_returns_exact_value(tariff: float) -> None:
    """validate_tariff returns the exact tariff value for positive inputs."""
    from energy_seer.app.validators import validate_tariff

    result = validate_tariff(tariff)
    assert result == tariff


class TestValidateFeatureWindow:
    def test_valid_window(self) -> None:
        from energy_seer.app.validators import validate_feature_window

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert validate_feature_window(values) == values

    def test_too_short_raises(self) -> None:
        from energy_seer.app.validators import validate_feature_window

        with pytest.raises(ValueError, match="at least"):
            validate_feature_window([1.0, 2.0], min_length=5)

    def test_custom_min_length(self) -> None:
        from energy_seer.app.validators import validate_feature_window

        assert validate_feature_window([1.0, 2.0], min_length=2) == [1.0, 2.0]

    def test_empty_raises(self) -> None:
        from energy_seer.app.validators import validate_feature_window

        with pytest.raises(ValueError):
            validate_feature_window([], min_length=1)


class TestValidateForecastLength:
    def test_valid_length(self) -> None:
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

    def test_max_boundary_accepted(self) -> None:
        from energy_seer.app.validators import validate_forecast_length

        assert validate_forecast_length(8760) == 8760


class TestValidateReadingsList:
    def test_valid_readings(self) -> None:
        from energy_seer.app.validators import validate_readings_list

        readings = [0.5, 1.0, 2.5]
        assert validate_readings_list(readings) == readings

    def test_negative_not_allowed_raises(self) -> None:
        from energy_seer.app.validators import validate_readings_list

        with pytest.raises(ValueError, match="negative"):
            validate_readings_list([-1.0, 2.0])

    def test_negative_allowed(self) -> None:
        from energy_seer.app.validators import validate_readings_list

        readings = [-1.0, 0.0, 1.0]
        assert validate_readings_list(readings, allow_negative=True) == readings

    def test_empty_list_raises(self) -> None:
        from energy_seer.app.validators import validate_readings_list

        with pytest.raises(ValueError):
            validate_readings_list([])
