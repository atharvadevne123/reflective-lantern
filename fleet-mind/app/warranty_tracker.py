"""Warranty tracking for fleet vehicles and components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class WarrantyRecord:
    vehicle_id: str
    component: str
    start_date: date
    expiry_date: date
    max_km: float | None = None
    current_km: float = 0.0

    def is_expired_by_date(self, today: date | None = None) -> bool:
        today = today or date.today()
        return today > self.expiry_date

    def is_expired_by_km(self) -> bool:
        if self.max_km is None:
            return False
        return self.current_km > self.max_km

    def is_expired(self, today: date | None = None) -> bool:
        return self.is_expired_by_date(today) or self.is_expired_by_km()

    def days_remaining(self, today: date | None = None) -> int:
        today = today or date.today()
        return (self.expiry_date - today).days

    def km_remaining(self) -> float | None:
        if self.max_km is None:
            return None
        return max(0.0, self.max_km - self.current_km)


def expiring_soon(
    records: list[WarrantyRecord],
    within_days: int = 30,
    today: date | None = None,
) -> list[WarrantyRecord]:
    """Return records expiring within the given number of days."""
    today = today or date.today()
    return [
        r for r in records
        if not r.is_expired(today) and r.days_remaining(today) <= within_days
    ]


def active_warranties(records: list[WarrantyRecord], today: date | None = None) -> list[WarrantyRecord]:
    """Return non-expired warranty records."""
    today = today or date.today()
    return [r for r in records if not r.is_expired(today)]


def warranty_coverage_rate(records: list[WarrantyRecord], today: date | None = None) -> float:
    """Return fraction of records that are still active."""
    if not records:
        return 0.0
    active = len(active_warranties(records, today))
    return active / len(records)
