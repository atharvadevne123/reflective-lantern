"""Maintenance cost optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepairOption:
    description: str
    cost: float
    downtime_hours: float
    reliability_gain: float


@dataclass
class CostModel:
    labour_rate_per_hour: float = 85.0
    downtime_cost_per_hour: float = 200.0
    preventive_discount: float = 0.20


def total_repair_cost(option: RepairOption, model: CostModel) -> float:
    """Return total cost including parts, labour, and downtime."""
    return option.cost + option.downtime_hours * model.downtime_cost_per_hour


def preventive_vs_reactive_saving(
    reactive_cost: float,
    preventive_cost: float,
) -> float:
    """Return cost saving achieved by choosing preventive over reactive maintenance."""
    return max(0.0, reactive_cost - preventive_cost)


def cost_per_reliability_unit(option: RepairOption, model: CostModel) -> float:
    """Return cost per unit of reliability gained (lower is better)."""
    total = total_repair_cost(option, model)
    if option.reliability_gain <= 0:
        return float("inf")
    return total / option.reliability_gain


def rank_options(options: list[RepairOption], model: CostModel) -> list[RepairOption]:
    """Return options sorted by cost-per-reliability-unit ascending."""
    return sorted(options, key=lambda o: cost_per_reliability_unit(o, model))


def annual_maintenance_budget(
    vehicle_count: int,
    avg_cost_per_vehicle: float,
    contingency_pct: float = 0.10,
) -> float:
    """Estimate annual maintenance budget with a contingency buffer."""
    base = vehicle_count * avg_cost_per_vehicle
    return base * (1 + contingency_pct)


def break_even_km(
    preventive_cost: float,
    reactive_cost: float,
    cost_per_km: float,
) -> float:
    """Return km at which preventive maintenance breaks even vs reactive."""
    if cost_per_km <= 0:
        return float("inf")
    saving = preventive_vs_reactive_saving(reactive_cost, preventive_cost)
    return saving / cost_per_km
