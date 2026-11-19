"""Tests for fleet-mind app/emissions_calculator.py."""

from __future__ import annotations

import pytest

from app.emissions_calculator import (
    FuelType,
    VehicleEmission,
    carbon_intensity_tier,
    co2_kg,
    emissions_per_km,
    fleet_co2_kg,
)


class TestCo2Kg:
    def test_petrol_emission(self) -> None:
        e = VehicleEmission("v1", FuelType.PETROL, fuel_consumed_litres=10.0)
        assert co2_kg(e) == pytest.approx(23.1, abs=0.01)

    def test_diesel_emission(self) -> None:
        e = VehicleEmission("v1", FuelType.DIESEL, fuel_consumed_litres=10.0)
        assert co2_kg(e) == pytest.approx(26.8, abs=0.01)

    def test_electric_zero_combustion(self) -> None:
        e = VehicleEmission("v1", FuelType.ELECTRIC, electricity_consumed_kwh=0.0)
        assert co2_kg(e) == pytest.approx(0.0)

    def test_electric_grid_emissions(self) -> None:
        e = VehicleEmission("v1", FuelType.ELECTRIC, electricity_consumed_kwh=100.0)
        result = co2_kg(e)
        assert result > 0.0


class TestFleetCo2Kg:
    def test_empty_fleet(self) -> None:
        assert fleet_co2_kg([]) == pytest.approx(0.0)

    def test_sums_emissions(self) -> None:
        emissions = [
            VehicleEmission("v1", FuelType.PETROL, fuel_consumed_litres=10.0),
            VehicleEmission("v2", FuelType.PETROL, fuel_consumed_litres=10.0),
        ]
        assert fleet_co2_kg(emissions) == pytest.approx(46.2, abs=0.1)


class TestEmissionsPerKm:
    def test_zero_km_returns_zero(self) -> None:
        assert emissions_per_km(100.0, 0.0) == pytest.approx(0.0)

    def test_calculates_gco2_per_km(self) -> None:
        # 10 kg CO2 over 100 km = 100 gCO2/km
        assert emissions_per_km(10.0, 100.0) == pytest.approx(100.0)


class TestCarbonIntensityTier:
    def test_tier_a(self) -> None:
        assert carbon_intensity_tier(30.0) == "A"

    def test_tier_b(self) -> None:
        assert carbon_intensity_tier(75.0) == "B"

    def test_tier_e(self) -> None:
        assert carbon_intensity_tier(250.0) == "E"
