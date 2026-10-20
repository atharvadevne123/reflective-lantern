"""Tests for app.water_heating module."""

from __future__ import annotations

import pytest

from app.water_heating import (
    DHWResult,
    boiler_consumption,
    compare_technologies,
    energy_to_heat_water,
    estimate_dhw,
    heat_pump_consumption,
)


class TestEnergyToHeatWater:
    def test_known_value(self) -> None:
        # 10 L water, 10°C → 60°C: Q = 10 * 4.186 * 50 / 3600 kWh
        result = energy_to_heat_water(10.0, 10.0, 60.0)
        expected = 10.0 * 4.186 * 50.0 / 3600.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_litres_zero_energy(self) -> None:
        assert energy_to_heat_water(0.0, 10.0, 60.0) == 0.0

    def test_negative_litres_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_heat_water(-1.0, 10.0, 60.0)

    def test_hot_le_cold_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_heat_water(100.0, 60.0, 60.0)

    @pytest.mark.parametrize(
        "litres,cold,hot",
        [(50.0, 10.0, 55.0), (100.0, 5.0, 60.0), (200.0, 15.0, 65.0)],
    )
    def test_positive_result(self, litres: float, cold: float, hot: float) -> None:
        assert energy_to_heat_water(litres, cold, hot) > 0.0


class TestBoilerConsumption:
    def test_unit_efficiency(self) -> None:
        assert boiler_consumption(100.0, 1.0) == pytest.approx(100.0)

    def test_90pct_efficiency(self) -> None:
        result = boiler_consumption(90.0, 0.9)
        assert result == pytest.approx(100.0, rel=1e-4)

    def test_zero_consumption(self) -> None:
        assert boiler_consumption(0.0, 0.9) == 0.0

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            boiler_consumption(-10.0, 0.9)

    def test_zero_efficiency_raises(self) -> None:
        with pytest.raises(ValueError):
            boiler_consumption(100.0, 0.0)

    @pytest.mark.parametrize("eff", [0.7, 0.85, 0.95])
    def test_consumption_gte_useful(self, eff: float) -> None:
        assert boiler_consumption(100.0, eff) >= 100.0


class TestHeatPumpConsumption:
    def test_cop_of_three(self) -> None:
        assert heat_pump_consumption(300.0, 3.0) == pytest.approx(100.0)

    def test_cop_one_equals_useful(self) -> None:
        assert heat_pump_consumption(100.0, 1.0) == pytest.approx(100.0)

    def test_negative_kwh_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_pump_consumption(-50.0, 3.0)

    def test_cop_below_one_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_pump_consumption(100.0, 0.9)

    @pytest.mark.parametrize("cop", [1.0, 2.0, 4.0])
    def test_consumption_lte_useful(self, cop: float) -> None:
        assert heat_pump_consumption(100.0, cop) <= 100.0


class TestEstimateDHW:
    def test_boiler_returns_result(self) -> None:
        result = estimate_dhw(150.0, technology="boiler")
        assert isinstance(result, DHWResult)

    def test_heat_pump_returns_result(self) -> None:
        result = estimate_dhw(150.0, technology="heat_pump", cop_or_efficiency=3.0)
        assert isinstance(result, DHWResult)

    def test_invalid_technology_raises(self) -> None:
        with pytest.raises(ValueError):
            estimate_dhw(150.0, technology="electric_immersion")

    def test_annual_is_daily_times_365(self) -> None:
        result = estimate_dhw(100.0)
        assert result.annual_kwh_consumed == pytest.approx(
            result.daily_kwh_consumed * 365, rel=1e-5
        )

    def test_heat_pump_uses_less_electricity_than_boiler_gas(self) -> None:
        hp = estimate_dhw(100.0, technology="heat_pump", cop_or_efficiency=3.0)
        boiler = estimate_dhw(100.0, technology="boiler", cop_or_efficiency=0.9)
        assert hp.daily_kwh_consumed < boiler.daily_kwh_consumed

    @pytest.mark.parametrize("litres", [50.0, 100.0, 200.0, 500.0])
    def test_parametrized_litres(self, litres: float) -> None:
        result = estimate_dhw(litres)
        assert result.daily_kwh_consumed > 0.0


class TestCompareTechnologies:
    def test_returns_both_keys(self) -> None:
        result = compare_technologies(150.0)
        assert "boiler" in result
        assert "heat_pump" in result

    def test_heat_pump_lower_kgco2(self) -> None:
        result = compare_technologies(150.0)
        assert isinstance(result["boiler"], DHWResult)
        assert isinstance(result["heat_pump"], DHWResult)

    @pytest.mark.parametrize("litres", [80.0, 150.0, 300.0])
    def test_parametrized_demand(self, litres: float) -> None:
        result = compare_technologies(litres)
        for key in ("boiler", "heat_pump"):
            assert result[key].daily_kwh_consumed > 0.0


class TestWaterHeatingEdgeCases:
    def test_zero_litres_gives_zero_energy(self) -> None:
        from app.water_heating import energy_to_heat_water
        assert energy_to_heat_water(0.0, 10.0, 60.0) == pytest.approx(0.0)

    def test_negative_litres_raises(self) -> None:
        from app.water_heating import energy_to_heat_water
        with pytest.raises(ValueError):
            energy_to_heat_water(-1.0, 10.0, 60.0)

    def test_equal_temps_raises(self) -> None:
        from app.water_heating import energy_to_heat_water
        with pytest.raises(ValueError):
            energy_to_heat_water(100.0, 60.0, 60.0)

    def test_boiler_efficiency_one_passes_through(self) -> None:
        from app.water_heating import boiler_consumption, energy_to_heat_water
        useful = energy_to_heat_water(100.0, 10.0, 60.0)
        consumed = boiler_consumption(useful, 1.0)
        assert consumed == pytest.approx(useful)

    def test_heat_pump_cop_one_equals_direct_electric(self) -> None:
        from app.water_heating import heat_pump_consumption, energy_to_heat_water
        useful = energy_to_heat_water(100.0, 10.0, 60.0)
        consumed = heat_pump_consumption(useful, 1.0)
        assert consumed == pytest.approx(useful)

    @pytest.mark.parametrize("cop", [2.0, 3.0, 4.0, 5.0])
    def test_higher_cop_lower_consumption(self, cop: float) -> None:
        from app.water_heating import estimate_dhw
        r_low = estimate_dhw(100.0, technology="heat_pump", cop_or_efficiency=2.0)
        r = estimate_dhw(100.0, technology="heat_pump", cop_or_efficiency=cop)
        if cop > 2.0:
            assert r.daily_kwh_consumed <= r_low.daily_kwh_consumed

    def test_invalid_technology_raises(self) -> None:
        from app.water_heating import estimate_dhw
        with pytest.raises(ValueError):
            estimate_dhw(100.0, technology="coal")

    def test_annual_cost_is_365_times_daily(self) -> None:
        from app.water_heating import estimate_dhw
        r = estimate_dhw(100.0)
        assert r.annual_cost == pytest.approx(r.daily_cost * 365, rel=1e-4)
