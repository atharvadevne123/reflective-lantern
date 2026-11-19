"""Fuel efficiency calculations for fleet vehicles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FuelMetrics:
    vehicle_id: str
    distance_km: float
    fuel_litres: float
    load_kg: float = 0.0


def litres_per_100km(distance_km: float, fuel_litres: float) -> float:
    """Return fuel consumption in L/100 km."""
    if distance_km <= 0:
        return 0.0
    return (fuel_litres / distance_km) * 100.0


def km_per_litre(distance_km: float, fuel_litres: float) -> float:
    """Return fuel economy in km/L."""
    if fuel_litres <= 0:
        return 0.0
    return distance_km / fuel_litres


def efficiency_score(l_per_100km: float, baseline: float = 10.0) -> float:
    """Return 0-1 efficiency score relative to a baseline consumption."""
    if baseline <= 0:
        return 0.0
    ratio = l_per_100km / baseline
    return max(0.0, min(1.0, 2.0 - ratio))


def load_adjusted_consumption(
    base_l_per_100km: float, load_kg: float, factor: float = 0.002
) -> float:
    """Return consumption adjusted for vehicle load."""
    return base_l_per_100km * (1 + factor * load_kg)


def fleet_average_efficiency(metrics: list[FuelMetrics]) -> float:
    """Return fleet-wide average fuel consumption in L/100 km."""
    total_distance = sum(m.distance_km for m in metrics)
    total_fuel = sum(m.fuel_litres for m in metrics)
    if total_distance <= 0:
        return 0.0
    return (total_fuel / total_distance) * 100.0


def co2_kg_from_litres(fuel_litres: float, emission_factor: float = 2.31) -> float:
    """Return CO2 emissions in kg (default factor for petrol)."""
    return fuel_litres * emission_factor
