"""Driver behavior analysis and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class DrivingEvent:
    driver_id: str
    event_type: str
    severity: float
    speed_kmh: float = 0.0


def harsh_event_rate(events: list[DrivingEvent], total_km: float) -> float:
    """Return harsh events per 100 km."""
    if total_km <= 0:
        return 0.0
    harsh = [e for e in events if e.event_type in {"harsh_brake", "harsh_accel", "harsh_corner"}]
    return (len(harsh) / total_km) * 100.0


def speeding_fraction(events: list[DrivingEvent], speed_limit_kmh: float) -> float:
    """Return fraction of speed-related events exceeding the speed limit."""
    if not events:
        return 0.0
    over = [e for e in events if e.speed_kmh > speed_limit_kmh]
    return len(over) / len(events)


def driver_score(
    events: list[DrivingEvent], total_km: float, speed_limit_kmh: float = 110.0
) -> float:
    """Return 0-100 driver score (higher is safer)."""
    if total_km <= 0:
        return 100.0
    her = harsh_event_rate(events, total_km)
    sf = speeding_fraction(events, speed_limit_kmh)
    score = 100.0 - (her * 5.0) - (sf * 30.0)
    return max(0.0, min(100.0, score))


def severity_stats(events: list[DrivingEvent]) -> dict[str, float]:
    """Return mean and std deviation of event severities."""
    severities = [e.severity for e in events]
    if not severities:
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {
        "mean": mean(severities),
        "std": stdev(severities) if len(severities) > 1 else 0.0,
        "max": max(severities),
    }


def rank_drivers(driver_scores: dict[str, float]) -> list[tuple[str, float]]:
    """Return drivers sorted from safest (highest score) to riskiest."""
    return sorted(driver_scores.items(), key=lambda x: x[1], reverse=True)
