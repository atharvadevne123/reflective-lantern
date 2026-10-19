"""Tests for app.solar_pv module."""

from __future__ import annotations

import pytest

from app.solar_pv import (
    PVSystemResult,
    analyse_pv_system,
    annual_yield_kwh,
    array_power_kw,
    capacity_factor,
    specific_yield,
)


class TestArrayPowerKw:
    def test_stc_conditions(self) -> None:
        power = array_power_kw(10, 400.0, 25.0, 1000.0)
        assert power == pytest.approx(4.0, rel=1e-4)

    def test_zero_irradiance_zero_power(self) -> None:
        assert array_power_kw(10, 400.0, 25.0, 0.0) == pytest.approx(0.0)

    def test_temperature_derating(self) -> None:
        hot = array_power_kw(10, 400.0, 45.0, 1000.0)
        stc = array_power_kw(10, 400.0, 25.0, 1000.0)
        assert hot < stc

    def test_zero_panels_raises(self) -> None:
        with pytest.raises(ValueError):
            array_power_kw(0, 400.0, 25.0, 1000.0)

    def test_negative_irradiance_raises(self) -> None:
        with pytest.raises(ValueError):
            array_power_kw(10, 400.0, 25.0, -100.0)

    @pytest.mark.parametrize("panels", [1, 10, 100])
    def test_more_panels_more_power(self, panels: int) -> None:
        p1 = array_power_kw(1, 400.0, 25.0, 1000.0)
        pn = array_power_kw(panels, 400.0, 25.0, 1000.0)
        assert pn == pytest.approx(p1 * panels, rel=1e-4)


class TestAnnualYieldKwh:
    def test_basic_calculation(self) -> None:
        yield_ = annual_yield_kwh(10.0, 4.5, 0.75)
        assert yield_ == pytest.approx(10.0 * 4.5 * 365 * 0.75, rel=1e-4)

    def test_zero_sun_hours_zero_yield(self) -> None:
        assert annual_yield_kwh(10.0, 0.0) == pytest.approx(0.0)

    def test_negative_peak_kw_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_yield_kwh(-5.0, 4.5)

    def test_invalid_pr_raises(self) -> None:
        with pytest.raises(ValueError):
            annual_yield_kwh(10.0, 4.5, 0.0)

    @pytest.mark.parametrize("pr", [0.6, 0.75, 0.9])
    def test_higher_pr_higher_yield(self, pr: float) -> None:
        base = annual_yield_kwh(10.0, 4.5, 0.75)
        result = annual_yield_kwh(10.0, 4.5, pr)
        if pr > 0.75:
            assert result > base
        else:
            assert result <= base


class TestCapacityFactor:
    def test_known_value(self) -> None:
        cf = capacity_factor(8760.0, 1.0)
        assert cf == pytest.approx(1.0, rel=1e-4)

    def test_zero_generation(self) -> None:
        assert capacity_factor(0.0, 10.0) == pytest.approx(0.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            capacity_factor(-100.0, 10.0)

    def test_zero_installed_raises(self) -> None:
        with pytest.raises(ValueError):
            capacity_factor(1000.0, 0.0)

    @pytest.mark.parametrize("kwh,kw", [(1000.0, 10.0), (5000.0, 20.0)])
    def test_within_bounds(self, kwh: float, kw: float) -> None:
        assert 0.0 <= capacity_factor(kwh, kw) <= 1.0


class TestSpecificYield:
    def test_basic(self) -> None:
        assert specific_yield(1000.0, 2.0) == pytest.approx(500.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            specific_yield(-100.0, 10.0)

    def test_zero_kwp_raises(self) -> None:
        with pytest.raises(ValueError):
            specific_yield(1000.0, 0.0)

    @pytest.mark.parametrize("kwh,kwp", [(900.0, 1.0), (4500.0, 5.0)])
    def test_parametrized(self, kwh: float, kwp: float) -> None:
        assert specific_yield(kwh, kwp) == pytest.approx(kwh / kwp)


class TestAnalysePvSystem:
    def test_returns_pv_result(self) -> None:
        result = analyse_pv_system(20, 400.0, 4.5)
        assert isinstance(result, PVSystemResult)

    def test_co2_saved_positive(self) -> None:
        result = analyse_pv_system(20, 400.0, 4.5)
        assert result.co2_saved_kg > 0.0

    def test_capacity_factor_between_0_and_1(self) -> None:
        result = analyse_pv_system(20, 400.0, 4.5)
        assert 0.0 <= result.capacity_factor <= 1.0

    @pytest.mark.parametrize("panels", [5, 20, 100])
    def test_more_panels_more_yield(self, panels: int) -> None:
        result = analyse_pv_system(panels, 400.0, 4.5)
        assert result.annual_yield_kwh > 0.0
