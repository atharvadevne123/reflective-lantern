"""Tests for app.geothermal module."""

from __future__ import annotations

import pytest

from app.geothermal import (
    GeothermalPlantResult,
    GeothermalResource,
    carnot_efficiency,
    characterise_resource,
    geothermal_plant,
    heat_flow,
    reservoir_temperature,
    thermal_power_from_flow,
)


class TestReservoirTemperature:
    def test_surface(self) -> None:
        assert reservoir_temperature(0.0, surface_temperature_c=15.0) == pytest.approx(15.0)

    def test_two_km_depth(self) -> None:
        t = reservoir_temperature(2000.0, surface_temperature_c=15.0, gradient_c_per_km=25.0)
        assert t == pytest.approx(65.0, rel=1e-4)

    def test_negative_depth_raises(self) -> None:
        with pytest.raises(ValueError):
            reservoir_temperature(-100.0)

    @pytest.mark.parametrize("depth,expected", [(1000.0, 40.0), (3000.0, 90.0)])
    def test_parametrized(self, depth: float, expected: float) -> None:
        t = reservoir_temperature(depth, surface_temperature_c=15.0, gradient_c_per_km=25.0)
        assert t == pytest.approx(expected, rel=1e-3)


class TestHeatFlow:
    def test_basic(self) -> None:
        hf = heat_flow(2.5, 0.025)
        assert hf == pytest.approx(0.0625, rel=1e-4)

    def test_zero_conductivity_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_flow(0.0, 0.025)

    def test_zero_gradient_raises(self) -> None:
        with pytest.raises(ValueError):
            heat_flow(2.5, 0.0)


class TestCarnotEfficiency:
    def test_basic(self) -> None:
        eff = carnot_efficiency(150.0, 25.0)
        assert 0 < eff < 1

    def test_equal_temps_raises(self) -> None:
        with pytest.raises(ValueError):
            carnot_efficiency(100.0, 100.0)

    def test_source_below_rejection_raises(self) -> None:
        with pytest.raises(ValueError):
            carnot_efficiency(20.0, 100.0)

    def test_higher_source_better_efficiency(self) -> None:
        eff1 = carnot_efficiency(100.0, 25.0)
        eff2 = carnot_efficiency(200.0, 25.0)
        assert eff2 > eff1

    @pytest.mark.parametrize("source,sink", [(100.0, 25.0), (200.0, 30.0), (300.0, 15.0)])
    def test_returns_fraction(self, source: float, sink: float) -> None:
        eff = carnot_efficiency(source, sink)
        assert 0 < eff < 1


class TestThermalPowerFromFlow:
    def test_basic(self) -> None:
        power = thermal_power_from_flow(1.0, 100.0, 60.0)
        assert power == pytest.approx(1.0 * 4.186 * 40.0, rel=1e-3)

    def test_negative_flow_raises(self) -> None:
        with pytest.raises(ValueError):
            thermal_power_from_flow(-1.0, 100.0, 60.0)

    def test_inlet_not_above_outlet_raises(self) -> None:
        with pytest.raises(ValueError):
            thermal_power_from_flow(1.0, 60.0, 100.0)

    def test_equal_temps_raises(self) -> None:
        with pytest.raises(ValueError):
            thermal_power_from_flow(1.0, 80.0, 80.0)


class TestGeothermalPlant:
    def test_returns_dataclass(self) -> None:
        result = geothermal_plant(150.0)
        assert isinstance(result, GeothermalPlantResult)

    def test_positive_electrical_power(self) -> None:
        result = geothermal_plant(150.0)
        assert result.electrical_power_kw > 0

    def test_source_below_rejection_raises(self) -> None:
        with pytest.raises(ValueError):
            geothermal_plant(20.0, rejection_temp_c=50.0)

    def test_zero_thermal_power_raises(self) -> None:
        with pytest.raises(ValueError):
            geothermal_plant(150.0, thermal_power_kw=0.0)

    def test_total_eff_less_than_carnot(self) -> None:
        result = geothermal_plant(150.0, plant_efficiency_fraction=0.5)
        assert result.thermal_efficiency < result.carnot_efficiency

    def test_co2_saved_positive(self) -> None:
        result = geothermal_plant(150.0)
        assert result.co2_saved_kg > 0


class TestCharacteriseResource:
    def test_returns_dataclass(self) -> None:
        resource = characterise_resource(2000.0)
        assert isinstance(resource, GeothermalResource)

    def test_temperature_matches_formula(self) -> None:
        resource = characterise_resource(1000.0, surface_temperature_c=15.0)
        expected = reservoir_temperature(1000.0, 15.0)
        assert resource.temperature_c == pytest.approx(expected)

    def test_deeper_resource_hotter(self) -> None:
        r1 = characterise_resource(1000.0)
        r2 = characterise_resource(3000.0)
        assert r2.temperature_c > r1.temperature_c


class TestGeothermalEdgeCases:
    @pytest.mark.parametrize(
        "depth,surface_temp,gradient",
        [
            (500.0, 10.0, 20.0),
            (1000.0, 20.0, 30.0),
            (5000.0, 5.0, 25.0),
        ],
    )
    def test_reservoir_temperature_increases_with_depth(
        self, depth: float, surface_temp: float, gradient: float
    ) -> None:
        t = reservoir_temperature(
            depth, surface_temperature_c=surface_temp, gradient_c_per_km=gradient
        )
        assert t > surface_temp

    @pytest.mark.parametrize(
        "flow,inlet,outlet",
        [
            (1.0, 100.0, 50.0),
            (2.0, 150.0, 80.0),
            (0.5, 90.0, 40.0),
        ],
    )
    def test_thermal_power_scales_with_flow(self, flow: float, inlet: float, outlet: float) -> None:
        p1 = thermal_power_from_flow(flow, inlet, outlet)
        p2 = thermal_power_from_flow(flow * 2, inlet, outlet)
        assert pytest.approx(p2, rel=1e-3) == p1 * 2

    def test_characterise_resource_temperature_in_range(self) -> None:
        resource = characterise_resource(2000.0)
        assert resource.temperature_c > 0.0

    def test_plant_efficiency_one_half_reduces_output(self) -> None:
        full = geothermal_plant(200.0, plant_efficiency_fraction=1.0)
        half = geothermal_plant(200.0, plant_efficiency_fraction=0.5)
        assert half.electrical_power_kw < full.electrical_power_kw
