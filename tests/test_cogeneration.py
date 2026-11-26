"""Tests for app.cogeneration module."""

from __future__ import annotations

import pytest

from app.cogeneration import (
    CHPResult,
    analyse_chp,
    chp_electrical_efficiency,
    chp_thermal_efficiency,
    co2_savings,
    total_cogeneration_efficiency,
)


class TestCHPElectricalEfficiency:
    def test_basic(self) -> None:
        eff = chp_electrical_efficiency(300.0, 1000.0)
        assert eff == pytest.approx(0.3, rel=1e-4)

    def test_zero_output_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_electrical_efficiency(0.0, 1000.0)

    def test_zero_input_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_electrical_efficiency(300.0, 0.0)

    def test_output_above_input_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_electrical_efficiency(1100.0, 1000.0)

    @pytest.mark.parametrize(
        "elec,fuel,expected",
        [
            (250.0, 1000.0, 0.25),
            (400.0, 1000.0, 0.40),
        ],
    )
    def test_parametrized(self, elec: float, fuel: float, expected: float) -> None:
        assert chp_electrical_efficiency(elec, fuel) == pytest.approx(expected, rel=1e-4)


class TestCHPThermalEfficiency:
    def test_basic(self) -> None:
        eff = chp_thermal_efficiency(500.0, 1000.0)
        assert eff == pytest.approx(0.5, rel=1e-4)

    def test_zero_heat_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_thermal_efficiency(0.0, 1000.0)

    def test_zero_fuel_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_thermal_efficiency(500.0, 0.0)

    def test_heat_above_fuel_raises(self) -> None:
        with pytest.raises(ValueError):
            chp_thermal_efficiency(1100.0, 1000.0)


class TestTotalCogenerationEfficiency:
    def test_basic(self) -> None:
        eff = total_cogeneration_efficiency(300.0, 500.0, 1000.0)
        assert eff == pytest.approx(0.8, rel=1e-4)

    def test_zero_electrical_raises(self) -> None:
        with pytest.raises(ValueError):
            total_cogeneration_efficiency(0.0, 500.0, 1000.0)

    def test_outputs_exceed_input_raises(self) -> None:
        with pytest.raises(ValueError):
            total_cogeneration_efficiency(700.0, 700.0, 1000.0)

    @pytest.mark.parametrize(
        "elec,heat,fuel",
        [
            (300.0, 500.0, 1000.0),
            (200.0, 600.0, 900.0),
        ],
    )
    def test_total_eff_in_range(self, elec: float, heat: float, fuel: float) -> None:
        eff = total_cogeneration_efficiency(elec, heat, fuel)
        assert 0 < eff <= 1


class TestCO2Savings:
    def test_positive_savings(self) -> None:
        savings = co2_savings(100.0, 200.0)
        assert savings > 0

    def test_negative_elec_raises(self) -> None:
        with pytest.raises(ValueError):
            co2_savings(-10.0, 100.0)

    def test_negative_heat_raises(self) -> None:
        with pytest.raises(ValueError):
            co2_savings(100.0, -10.0)

    def test_zero_energy_returns_value(self) -> None:
        savings = co2_savings(0.0, 0.0)
        assert savings == pytest.approx(0.0, abs=1e-4)


class TestAnalyseCHP:
    def test_returns_dataclass(self) -> None:
        result = analyse_chp(1000.0, 0.35, 0.45)
        assert isinstance(result, CHPResult)

    def test_efficiencies_sum_correct(self) -> None:
        result = analyse_chp(1000.0, 0.35, 0.45)
        assert result.total_efficiency == pytest.approx(0.80, rel=1e-4)

    def test_zero_fuel_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_chp(0.0, 0.35, 0.45)

    def test_efficiency_sum_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_chp(1000.0, 0.6, 0.5)

    def test_invalid_electrical_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_chp(1000.0, 0.0, 0.5)

    def test_invalid_thermal_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            analyse_chp(1000.0, 0.35, -0.1)

    def test_phr_positive(self) -> None:
        result = analyse_chp(1000.0, 0.35, 0.45)
        assert result.power_to_heat_ratio > 0

    def test_co2_savings_positive(self) -> None:
        result = analyse_chp(1000.0, 0.35, 0.45)
        assert result.annual_co2_savings_kg > 0

    @pytest.mark.parametrize(
        "elec_eff,therm_eff",
        [
            (0.25, 0.50),
            (0.35, 0.45),
            (0.40, 0.40),
        ],
    )
    def test_parametrized_efficiencies(self, elec_eff: float, therm_eff: float) -> None:
        result = analyse_chp(1000.0, elec_eff, therm_eff)
        assert result.electrical_efficiency == pytest.approx(elec_eff)
        assert result.thermal_efficiency == pytest.approx(therm_eff)


class TestCogenerationEdgeCases:
    @pytest.mark.parametrize(
        "fuel_input,elec_eff,therm_eff",
        [
            (500.0, 0.30, 0.50),
            (2000.0, 0.40, 0.40),
            (10000.0, 0.35, 0.45),
        ],
    )
    def test_analyse_chp_total_efficiency_bounded(
        self, fuel_input: float, elec_eff: float, therm_eff: float
    ) -> None:
        from app.cogeneration import analyse_chp

        result = analyse_chp(fuel_input, elec_eff, therm_eff)
        assert 0.0 < result.total_efficiency <= 1.0

    def test_heat_to_power_ratio_basic(self) -> None:
        from app.cogeneration import heat_to_power_ratio

        # (thermal_output_kw, electrical_output_kw)
        ratio = heat_to_power_ratio(450.0, 350.0)
        assert ratio == pytest.approx(450.0 / 350.0, rel=1e-3)

    def test_total_cogeneration_efficiency_sum(self) -> None:
        from app.cogeneration import total_cogeneration_efficiency

        # elec + heat / fuel
        total = total_cogeneration_efficiency(350.0, 450.0, 1000.0)
        assert total == pytest.approx(0.80, rel=1e-3)

    def test_co2_savings_positive(self) -> None:
        from app.cogeneration import co2_savings

        savings = co2_savings(1000.0, 0.35, 0.45)
        assert savings > 0.0

    @pytest.mark.parametrize(
        "elec_output,fuel_input",
        [
            (350.0, 1000.0),
            (140.0, 400.0),
            (700.0, 2000.0),
        ],
    )
    def test_chp_electrical_efficiency_ratio(self, elec_output: float, fuel_input: float) -> None:
        from app.cogeneration import chp_electrical_efficiency

        eff = chp_electrical_efficiency(elec_output, fuel_input)
        assert eff == pytest.approx(elec_output / fuel_input, rel=1e-3)
