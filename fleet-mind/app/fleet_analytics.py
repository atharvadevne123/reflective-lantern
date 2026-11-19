"""Fleet-wide aggregated analytics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class VehicleSummary:
    vehicle_id: str
    odometer_km: float
    maintenance_cost_ytd: float
    uptime_fraction: float
    anomaly_count: int
    fuel_efficiency_l_per_100km: float


def fleet_uptime(summaries: list[VehicleSummary]) -> float:
    """Return mean fleet uptime fraction."""
    if not summaries:
        return 0.0
    return mean(s.uptime_fraction for s in summaries)


def total_maintenance_cost(summaries: list[VehicleSummary]) -> float:
    """Return total YTD maintenance cost across the fleet."""
    return sum(s.maintenance_cost_ytd for s in summaries)


def cost_per_km(summaries: list[VehicleSummary]) -> float:
    """Return fleet-wide maintenance cost per km driven."""
    total_km = sum(s.odometer_km for s in summaries)
    total_cost = total_maintenance_cost(summaries)
    if total_km <= 0:
        return 0.0
    return total_cost / total_km


def top_anomaly_vehicles(summaries: list[VehicleSummary], n: int = 5) -> list[VehicleSummary]:
    """Return the n vehicles with the most anomaly events."""
    return sorted(summaries, key=lambda s: s.anomaly_count, reverse=True)[:n]


def efficiency_distribution(summaries: list[VehicleSummary]) -> dict[str, int]:
    """Bucket vehicles by L/100km into efficiency bands."""
    bands: dict[str, int] = {"<6": 0, "6-9": 0, "9-12": 0, ">12": 0}
    for s in summaries:
        v = s.fuel_efficiency_l_per_100km
        if v < 6:
            bands["<6"] += 1
        elif v < 9:
            bands["6-9"] += 1
        elif v < 12:
            bands["9-12"] += 1
        else:
            bands[">12"] += 1
    return bands


def fleet_health_index(summaries: list[VehicleSummary]) -> float:
    """Return 0-1 composite health index (uptime, efficiency, low anomalies)."""
    if not summaries:
        return 0.0
    avg_uptime = fleet_uptime(summaries)
    max_anomalies = max(s.anomaly_count for s in summaries) or 1
    avg_anomaly_norm = 1.0 - mean(s.anomaly_count / max_anomalies for s in summaries)
    avg_efficiency_norm = 1.0 - min(
        1.0, mean(s.fuel_efficiency_l_per_100km for s in summaries) / 20.0
    )
    return (avg_uptime + avg_anomaly_norm + avg_efficiency_norm) / 3.0
