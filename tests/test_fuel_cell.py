"""Tests for app.fuel_cell module."""

from __future__ import annotations

import math

import pytest

from app.fuel_cell import (
    FuelCellResult,
    activation_overpotential,
    cell_voltage,
    fuel_cell_efficiency,
    hydrogen_consumption_rate,
    nernst_voltage,
    ohmic_overpotential,
    operating_point,
    stack_power,
)


class TestNernstVoltage:
    def test_standard_conditions(self) -> None:
        v = nernst_voltage(298.15)
        assert 1.0 < v < 1.3

    def test_higher_temperature_reduces_voltage(self) -> None:
        v_low = nernst_voltage(298.15)
        v_high = nernst_voltage(353.15)
        assert v_high < v_low

    def test_zero_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            nernst_voltage(0.0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            nernst_voltage(-10.0)

    def test_zero_h2_pressure_raises(self) -> None:
        with pytest.raises(ValueError):
            nernst_voltage(353.15, partial_pressure_h2=0.0)

    @pytest.mark.parametrize("temp_k", [300.0, 353.15, 400.0])
    def test_returns_float(self, temp_k: float) -> None:
        assert isinstance(nernst_voltage(temp_k), float)


class TestActivationOverpotential:
    def test_positive_current(self) -> None:
        v = activation_overpotential(0.1)
        assert v > 0

    def test_zero_current_raises(self) -> None:
        with pytest.raises(ValueError):
            activation_overpotential(0.0)

    def test_negative_current_raises(self) -> None:
        with pytest.raises(ValueError):
            activation_overpotential(-0.1)

    def test_zero_exchange_current_raises(self) -> None:
        with pytest.raises(ValueError):
            activation_overpotential(0.1, exchange_current_density=0.0)


class TestOhmicOverpotential:
    def test_basic(self) -> None:
        assert ohmic_overpotential(100.0, 0.001) == pytest.approx(0.1, rel=1e-4)

    def test_zero_current(self) -> None:
        assert ohmic_overpotential(0.0, 0.001) == 0.0

    def test_negative_current_raises(self) -> None:
        with pytest.raises(ValueError):
            ohmic_overpotential(-1.0, 0.001)

    def test_negative_resistance_raises(self) -> None:
        with pytest.raises(ValueError):
            ohmic_overpotential(100.0, -0.001)


class TestCellVoltage:
    def test_no_losses(self) -> None:
        assert cell_voltage(1.2, 0.0, 0.0) == pytest.approx(1.2)

    def test_floored_at_zero(self) -> None:
        assert cell_voltage(0.5, 0.6, 0.2) == 0.0

    def test_with_losses(self) -> None:
        v = cell_voltage(1.2, 0.1, 0.05)
        assert v == pytest.approx(1.05, rel=1e-4)


class TestStackPower:
    def test_basic(self) -> None:
        p = stack_power(0.7, 100.0, 100)
        assert p == pytest.approx(7.0, rel=1e-4)

    def test_zero_voltage_raises(self) -> None:
        with pytest.raises(ValueError):
            stack_power(0.0, 100.0, 100)

    def test_zero_current_raises(self) -> None:
        with pytest.raises(ValueError):
            stack_power(0.7, 0.0, 100)

    def test_zero_cells_raises(self) -> None:
        with pytest.raises(ValueError):
            stack_power(0.7, 100.0, 0)

    @pytest.mark.parametrize("n_cells", [50, 100, 200])
    def test_scales_linearly_with_cells(self, n_cells: int) -> None:
        p_50 = stack_power(0.7, 100.0, 50)
        p = stack_power(0.7, 100.0, n_cells)
        assert p == pytest.approx(p_50 * n_cells / 50, rel=1e-4)


class TestFuelCellEfficiency:
    def test_hhv_reference(self) -> None:
        eff = fuel_cell_efficiency(1.0, use_hhv=True)
        assert 0 < eff <= 1

    def test_lhv_reference(self) -> None:
        eff = fuel_cell_efficiency(1.0, use_hhv=False)
        assert eff > fuel_cell_efficiency(1.0, use_hhv=True)

    def test_zero_voltage_raises(self) -> None:
        with pytest.raises(ValueError):
            fuel_cell_efficiency(0.0)

    def test_capped_at_one(self) -> None:
        assert fuel_cell_efficiency(2.0) == 1.0


class TestHydrogenConsumptionRate:
    def test_positive(self) -> None:
        rate = hydrogen_consumption_rate(10.0, 0.6)
        assert rate > 0

    def test_zero_power_raises(self) -> None:
        with pytest.raises(ValueError):
            hydrogen_consumption_rate(0.0, 0.6)

    def test_zero_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            hydrogen_consumption_rate(10.0, 0.0)


class TestOperatingPoint:
    def test_returns_dataclass(self) -> None:
        result = operating_point(100.0, 100)
        assert isinstance(result, FuelCellResult)

    def test_positive_power(self) -> None:
        result = operating_point(100.0, 100)
        assert result.power_kw > 0

    def test_stack_voltage_equals_cell_times_n(self) -> None:
        result = operating_point(100.0, 50)
        assert result.stack_voltage_v == pytest.approx(result.cell_voltage_v * 50, rel=1e-4)

    def test_zero_current_raises(self) -> None:
        with pytest.raises(ValueError):
            operating_point(0.0, 100)

    def test_zero_cells_raises(self) -> None:
        with pytest.raises(ValueError):
            operating_point(100.0, 0)

    def test_zero_area_raises(self) -> None:
        with pytest.raises(ValueError):
            operating_point(100.0, 100, active_area_cm2=0.0)

    @pytest.mark.parametrize("current", [50.0, 100.0, 200.0])
    def test_power_increases_with_current(self, current: float) -> None:
        r1 = operating_point(50.0, 100)
        r2 = operating_point(current, 100)
        if current > 50.0:
            assert r2.power_kw >= r1.power_kw


class TestFuelCellNewFunctions:
    def test_degradation_rate_per_hour_basic(self) -> None:
        from app.fuel_cell import degradation_rate_per_hour
        rate = degradation_rate_per_hour(1000.0, 5.0)
        assert rate == pytest.approx(5e-6, rel=1e-4)

    def test_degradation_rate_zero_hours_raises(self) -> None:
        from app.fuel_cell import degradation_rate_per_hour
        with pytest.raises(ValueError):
            degradation_rate_per_hour(0.0, 5.0)

    def test_water_production_matches_h2_flow(self) -> None:
        from app.fuel_cell import water_production_rate
        assert water_production_rate(0.01) == pytest.approx(0.01)

    def test_water_production_negative_raises(self) -> None:
        from app.fuel_cell import water_production_rate
        with pytest.raises(ValueError):
            water_production_rate(-0.001)

    def test_stack_voltage_scales_with_cells(self) -> None:
        from app.fuel_cell import stack_voltage
        v1 = stack_voltage(0.7, 10)
        v2 = stack_voltage(0.7, 20)
        assert v2 == pytest.approx(2 * v1, rel=1e-5)

    def test_tidal_energy_density_zero_speed_zero(self) -> None:
        from app.tidal_power import tidal_energy_density
        assert tidal_energy_density(0.0) == pytest.approx(0.0)

    def test_annual_tidal_energy_scales_with_capacity(self) -> None:
        from app.tidal_power import annual_tidal_energy_kwh
        e1 = annual_tidal_energy_kwh(100.0)
        e2 = annual_tidal_energy_kwh(200.0)
        assert e2 == pytest.approx(2 * e1, rel=1e-4)

    def test_tidal_resource_score_below_cutin_zero(self) -> None:
        from app.tidal_power import tidal_resource_score
        assert tidal_resource_score(0.3, cut_in_speed_m_s=0.5) == pytest.approx(0.0)

    def test_geothermal_gradient_increases_with_depth(self) -> None:
        from app.geothermal import geothermal_gradient
        t1 = geothermal_gradient(15.0, 1000.0)
        t2 = geothermal_gradient(15.0, 3000.0)
        assert t2 > t1

    def test_lcoe_decreases_with_more_generation(self) -> None:
        from app.geothermal import levelised_cost_of_energy
        lcoe_low = levelised_cost_of_energy(1e6, 50000, 5e6)
        lcoe_high = levelised_cost_of_energy(1e6, 50000, 10e6)
        assert lcoe_high < lcoe_low

    def test_battery_cycles_per_year_scales(self) -> None:
        from app.peak_shaving import battery_cycles_per_year
        c1 = battery_cycles_per_year(25.0, 100.0)
        c2 = battery_cycles_per_year(50.0, 100.0)
        assert c2 == pytest.approx(2 * c1, rel=1e-4)

    @pytest.mark.parametrize("reduction", [0.0, 50.0, 200.0])
    def test_revenue_from_peak_shaving_non_negative(self, reduction: float) -> None:
        from app.peak_shaving import revenue_from_peak_shaving
        assert revenue_from_peak_shaving(reduction, 15.0) >= 0.0
