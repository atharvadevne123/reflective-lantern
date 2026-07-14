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


def mode_label(mode: RunMode) -> str:
    """Return a short human-readable label for a RunMode value.

    Args:
        mode: A RunMode enum value.

    Returns:
        "Innovation Mode" for INNOVATION, "Improvement Mode" otherwise.
    """
    return "Innovation Mode" if mode == RunMode.INNOVATION else "Improvement Mode"


__all__ = [
    "RunMode",
    "determine_mode",
    "is_innovation_day",
    "next_innovation_day",
    "days_until_next_innovation",
    "mode_label",
]
