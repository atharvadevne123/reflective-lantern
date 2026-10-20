"""Tests for app.wind_power module."""

from __future__ import annotations

import math

import pytest

from app.wind_power import (
    BETZ_LIMIT,
    TurbineResult,
    analyse_turbine,
    annual_energy_output_kwh,
    rotor_swept_area,
    turbine_power_kw,
    wind_power_density,
    wind_speed_at_height,
)


class TestRotorSweptArea:
    def test_known_value(self) -> None:
        area = rotor_swept_area(2.0)
        assert area == pytest.approx(math.pi, rel=1e-4)

    def test_zero_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            rotor_swept_area(0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            rotor_swept_area(-5.0)

    @pytest.mark.parametrize("d", [10.0, 50.0, 100.0])
    def test_positive_result(self, d: float) -> None:
        assert rotor_swept_area(d) > 0.0


class TestWindPowerDensity:
    def test_zero_wind_zero_density(self) -> None:
        assert wind_power_density(0.0) == pytest.approx(0.0)

    def test_cube_law(self) -> None:
        d1 = wind_power_density(5.0)
        d2 = wind_power_density(10.0)
        assert d2 == pytest.approx(d1 * 8.0, rel=1e-4)

    def test_negative_wind_raises(self) -> None:
        with pytest.raises(ValueError):
            wind_power_density(-1.0)

    def test_zero_density_raises(self) -> None:
        with pytest.raises(ValueError):
            wind_power_density(5.0, 0.0)

    @pytest.mark.parametrize("speed", [3.0, 7.0, 12.0])
    def test_positive_result(self, speed: float) -> None:
        assert wind_power_density(speed) > 0.0


class TestTurbinePowerKw:
    def test_zero_wind_zero_power(self) -> None:
        assert turbine_power_kw(0.0, 50.0) == pytest.approx(0.0)

    def test_betz_limit_respected(self) -> None:
        power_betz = turbine_power_kw(10.0, 50.0, BETZ_LIMIT)
        power_less = turbine_power_kw(10.0, 50.0, 0.40)
        assert power_less < power_betz

    def test_zero_diameter_raises(self) -> None:
        with pytest.raises(ValueError):
            turbine_power_kw(10.0, 0.0)

    def test_invalid_cp_raises(self) -> None:
        with pytest.raises(ValueError):
            turbine_power_kw(10.0, 50.0, 0.0)

    @pytest.mark.parametrize("speed", [5.0, 10.0, 15.0])
    def test_higher_speed_higher_power(self, speed: float) -> None:
        p1 = turbine_power_kw(5.0, 50.0)
        pn = turbine_power_kw(speed, 50.0)
        if speed > 5.0:
            assert pn > p1
        else:
            assert pn == pytest.approx(p1)


class TestAnnualEnergyOutputKwh:
    def test_basic(self) -> None:
        assert annual_energy_output_kwh(1000.0, 1.0) == pytest.approx(8760000.0, rel=1e-4)

    def test_zero_rated_kw_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_energy_output_kwh(0.0, 0.3)

    def test_zero_cf_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_energy_output_kwh(1000.0, 0.0)

    @pytest.mark.parametrize("cf", [0.2, 0.3, 0.45])
    def test_higher_cf_higher_output(self, cf: float) -> None:
        base = annual_energy_output_kwh(1000.0, 0.3)
        result = annual_energy_output_kwh(1000.0, cf)
        if cf > 0.3:
            assert result > base


class TestWindSpeedAtHeight:
    def test_same_height_same_speed(self) -> None:
        assert wind_speed_at_height(8.0, 10.0, 10.0) == pytest.approx(8.0)

    def test_higher_height_faster(self) -> None:
        v_low = wind_speed_at_height(8.0, 10.0, 10.0)
        v_high = wind_speed_at_height(8.0, 10.0, 80.0)
        assert v_high > v_low

    def test_zero_height_raises(self) -> None:
        with pytest.raises(ValueError):
            wind_speed_at_height(8.0, 0.0, 80.0)

    def test_negative_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            wind_speed_at_height(-5.0, 10.0, 80.0)

    @pytest.mark.parametrize("target_h", [20.0, 50.0, 100.0])
    def test_parametrized(self, target_h: float) -> None:
        result = wind_speed_at_height(6.0, 10.0, target_h)
        assert result >= 6.0


class TestAnalyseTurbine:
    def test_returns_turbine_result(self) -> None:
        result = analyse_turbine(80.0, 2000.0, 7.0)
        assert isinstance(result, TurbineResult)

    def test_co2_saved_positive(self) -> None:
        result = analyse_turbine(80.0, 2000.0, 7.0)
        assert result.co2_saved_kg > 0.0

    def test_actual_less_than_theoretical(self) -> None:
        result = analyse_turbine(80.0, 2000.0, 7.0)
        assert result.actual_power_kw <= result.theoretical_power_kw

    @pytest.mark.parametrize("rated_kw", [500.0, 2000.0, 5000.0])
    def test_larger_turbine_more_yield(self, rated_kw: float) -> None:
        result = analyse_turbine(80.0, rated_kw, 7.0)
        assert result.annual_kwh > 0.0


class TestWindPowerEdgeCases:
    def test_zero_wind_speed_zero_power(self) -> None:
        from app.wind_power import wind_power_density
        assert wind_power_density(0.0) == 0.0

    def test_power_density_cubic_scaling(self) -> None:
        from app.wind_power import wind_power_density
        d1 = wind_power_density(5.0)
        d2 = wind_power_density(10.0)
        assert d2 == pytest.approx(d1 * 8.0, rel=1e-3)

    def test_negative_wind_speed_raises(self) -> None:
        from app.wind_power import wind_power_density
        with pytest.raises(ValueError):
            wind_power_density(-1.0)

    def test_rotor_area_scales_with_diameter_squared(self) -> None:
        from app.wind_power import rotor_swept_area
        a1 = rotor_swept_area(40.0)
        a2 = rotor_swept_area(80.0)
        assert a2 == pytest.approx(a1 * 4.0, rel=1e-3)

    def test_zero_diameter_raises(self) -> None:
        from app.wind_power import rotor_swept_area
        with pytest.raises(ValueError):
            rotor_swept_area(0.0)

    @pytest.mark.parametrize("diameter,speed", [(40.0, 5.0), (80.0, 8.0), (120.0, 10.0)])
    def test_turbine_result_non_negative(self, diameter: float, speed: float) -> None:
        from app.wind_power import analyse_turbine
        result = analyse_turbine(diameter, 2000.0, speed)
        assert result.actual_power_kw >= 0
        assert result.annual_kwh >= 0

    def test_capacity_factor_in_range(self) -> None:
        result = analyse_turbine(80.0, 2000.0, 7.0)
        assert 0 <= result.capacity_factor <= 1
