"""Vehicle emissions calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FuelType(StrEnum):
    PETROL = "petrol"
    DIESEL = "diesel"
    LPG = "lpg"
    CNG = "cng"
    ELECTRIC = "electric"


EMISSION_FACTORS_KG_PER_LITRE: dict[str, float] = {
    FuelType.PETROL: 2.31,
    FuelType.DIESEL: 2.68,
    FuelType.LPG: 1.51,
    FuelType.CNG: 2.04,
    FuelType.ELECTRIC: 0.0,
}

ELECTRICITY_EMISSION_KG_PER_KWH = 0.233


@dataclass
class VehicleEmission:
    vehicle_id: str
    fuel_type: FuelType
    fuel_consumed_litres: float = 0.0
    electricity_consumed_kwh: float = 0.0


def co2_kg(
    emission: VehicleEmission, grid_intensity: float = ELECTRICITY_EMISSION_KG_PER_KWH
) -> float:
    """Return CO2 emissions in kg for a vehicle emission record."""
    factor = EMISSION_FACTORS_KG_PER_LITRE.get(emission.fuel_type, 0.0)
    combustion_co2 = emission.fuel_consumed_litres * factor
    electric_co2 = emission.electricity_consumed_kwh * grid_intensity
    return combustion_co2 + electric_co2


def fleet_co2_kg(emissions: list[VehicleEmission]) -> float:
    """Return total fleet CO2 in kg."""
    return sum(co2_kg(e) for e in emissions)


def emissions_per_km(total_co2_kg: float, total_km: float) -> float:
    """Return grams of CO2 per km (standard reporting unit)."""
    if total_km <= 0:
        return 0.0
    return (total_co2_kg / total_km) * 1000.0


def carbon_intensity_tier(gco2_per_km: float) -> str:
    """Classify gCO2/km into tier."""
    if gco2_per_km < 50:
        return "A"
    if gco2_per_km < 100:
        return "B"
    if gco2_per_km < 150:
        return "C"
    if gco2_per_km < 200:
        return "D"
    return "E"
