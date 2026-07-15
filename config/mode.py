"""Determine the daily run mode (IMPROVEMENT or INNOVATION)."""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum

from config.constants import INNOVATION_DAY_RANGES, INNOVATION_WEEKDAY


class RunMode(StrEnum):
    """Possible Reflective Lantern run modes.

    IMPROVEMENT: Standard Mon–Fri 60-commit code improvement pass.
    INNOVATION: Full project creation on qualifying Wednesdays (days 8–14 or 22–28).
    """

    IMPROVEMENT = "IMPROVEMENT"
    INNOVATION = "INNOVATION"


def determine_mode(target_date: date | None = None) -> RunMode:
    """Return the run mode for *target_date* (defaults to today).

    INNOVATION fires on Wednesday (isoweekday 3) when the day-of-month
    falls in one of the configured ranges (8-14 or 22-28).
    All other weekdays are IMPROVEMENT.
    """
    d = target_date or date.today()
    if d.isoweekday() == INNOVATION_WEEKDAY and any(
        lo <= d.day <= hi for lo, hi in INNOVATION_DAY_RANGES
    ):
        return RunMode.INNOVATION
    return RunMode.IMPROVEMENT


def is_innovation_day(target_date: date | None = None) -> bool:
    """Return True if *target_date* is an innovation day."""
    return determine_mode(target_date) == RunMode.INNOVATION


def next_innovation_day(after: date | None = None) -> date:
    """Return the next innovation day strictly after *after* (defaults to today).

    Searches up to 365 days forward. Raises RuntimeError if none found
    (should never happen given the fixed monthly cadence).
    """
    start = (after or date.today()) + timedelta(days=1)
    for offset in range(365):
        candidate = start + timedelta(days=offset)
        if is_innovation_day(candidate):
            return candidate
    raise RuntimeError("No innovation day found within 365 days")


def days_until_next_innovation(after: date | None = None) -> int:
    """Return how many days until the next innovation day."""
    ref = after or date.today()
    return (next_innovation_day(ref) - ref).days


__all__ = ["RunMode", "determine_mode", "is_innovation_day", "next_innovation_day", "days_until_next_innovation"]


def upcoming_innovation_days(count: int = 5, after: date | None = None) -> list[date]:
    """Return the next *count* innovation days after *after* (defaults to today).

    Args:
        count: Number of future innovation days to compute.
        after: Starting reference date (defaults to today).

    Returns:
        Sorted list of upcoming innovation dates.
    """
    results: list[date] = []
    current = after or date.today()
    for _ in range(count):
        current = next_innovation_day(current)
        results.append(current)
    return results


def mode_schedule(weeks: int = 4, start: date | None = None) -> list[dict[str, object]]:
    """Return the run-mode schedule for the next *weeks* weeks.

    Args:
        weeks: Number of weeks to project forward.
        start: Starting date (defaults to today).

    Returns:
        List of dicts with 'date' (ISO string) and 'mode' keys for each weekday.
    """
    ref = start or date.today()
    schedule: list[dict[str, object]] = []
    for offset in range(weeks * 7):
        d = ref + timedelta(days=offset)
        if d.isoweekday() <= 5:  # Mon–Fri only
            schedule.append({"date": d.isoformat(), "mode": str(determine_mode(d))})
    return schedule
