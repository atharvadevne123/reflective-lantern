"""Vehicle risk scoring based on telemetry and maintenance history."""

from __future__ import annotations

from dataclasses import dataclass

WEIGHT_ANOMALY = 0.35
WEIGHT_OVERDUE = 0.30
WEIGHT_AGE = 0.20
WEIGHT_MILEAGE = 0.15


@dataclass
class RiskInput:
    vehicle_id: str
    anomaly_score: float
    days_maintenance_overdue: int
    vehicle_age_years: float
    odometer_km: float
    max_age_years: float = 15.0
    max_odometer_km: float = 500_000.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def component_scores(inp: RiskInput) -> dict[str, float]:
    """Return individual risk component scores (0-1 each)."""
    anomaly = _clamp(inp.anomaly_score)
    overdue = _clamp(inp.days_maintenance_overdue / 90.0)
    age = _clamp(inp.vehicle_age_years / inp.max_age_years)
    mileage = _clamp(inp.odometer_km / inp.max_odometer_km)
    return {
        "anomaly": anomaly,
        "overdue_maintenance": overdue,
        "age": age,
        "mileage": mileage,
    }


def composite_risk_score(inp: RiskInput) -> float:
    """Return weighted composite risk score (0-1)."""
    c = component_scores(inp)
    return (
        WEIGHT_ANOMALY * c["anomaly"]
        + WEIGHT_OVERDUE * c["overdue_maintenance"]
        + WEIGHT_AGE * c["age"]
        + WEIGHT_MILEAGE * c["mileage"]
    )


def risk_tier(score: float) -> str:
    """Classify score into LOW / MEDIUM / HIGH / CRITICAL tier."""
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def fleet_risk_summary(inputs: list[RiskInput]) -> dict[str, int | float]:
    """Return fleet-wide risk summary statistics."""
    if not inputs:
        return {"count": 0, "mean_score": 0.0, "critical_count": 0, "high_count": 0}
    scores = [composite_risk_score(i) for i in inputs]
    tiers = [risk_tier(s) for s in scores]
    return {
        "count": len(inputs),
        "mean_score": sum(scores) / len(scores),
        "critical_count": tiers.count("CRITICAL"),
        "high_count": tiers.count("HIGH"),
    }
