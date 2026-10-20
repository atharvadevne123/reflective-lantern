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
