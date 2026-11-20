"""Aggregate and summarise fleet telemetry events."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date


@dataclass
class FleetEvent:
    vehicle_id: str
    event_type: str
    severity: str
    occurred_on: date
    value: float = 0.0


def count_by_type(events: list[FleetEvent]) -> dict[str, int]:
    """Return count of events grouped by type."""
    return dict(Counter(e.event_type for e in events))


def count_by_severity(events: list[FleetEvent]) -> dict[str, int]:
    """Return count of events grouped by severity."""
    return dict(Counter(e.severity for e in events))


def events_per_vehicle(events: list[FleetEvent]) -> dict[str, int]:
    """Return number of events per vehicle."""
    return dict(Counter(e.vehicle_id for e in events))


def events_in_window(
    events: list[FleetEvent],
    start: date,
    end: date,
) -> list[FleetEvent]:
    """Return events that occurred within [start, end] inclusive."""
    return [e for e in events if start <= e.occurred_on <= end]


def critical_event_rate(events: list[FleetEvent]) -> float:
    """Return fraction of events with critical severity."""
    if not events:
        return 0.0
    critical = sum(1 for e in events if e.severity == "critical")
    return critical / len(events)


def daily_event_counts(events: list[FleetEvent]) -> dict[str, int]:
    """Return event count per day (ISO date string keys)."""
    counts: dict[str, int] = defaultdict(int)
    for e in events:
        counts[e.occurred_on.isoformat()] += 1
    return dict(counts)
