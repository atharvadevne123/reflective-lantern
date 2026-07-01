"""Determine the daily run mode (IMPROVEMENT or INNOVATION)."""

from __future__ import annotations

from datetime import date
from enum import Enum

from config.constants import INNOVATION_DAY_RANGES, INNOVATION_WEEKDAY


class RunMode(str, Enum):
    """Possible Reflective Lantern run modes."""

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
