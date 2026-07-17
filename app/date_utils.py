"""Date and time utility helpers for Watt-Guard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def round_to_hour(dt: datetime) -> datetime:
    """Round *dt* down to the nearest hour.

    Args:
        dt: Input datetime (any timezone).

    Returns:
        Datetime with minutes, seconds, and microseconds zeroed out.
    """
    return dt.replace(minute=0, second=0, microsecond=0)


def hours_between(start: datetime, end: datetime) -> int:
    """Return the number of whole hours between *start* and *end*.

    Args:
        start: Earlier datetime.
        end: Later datetime.

    Returns:
        Non-negative integer count of whole hours; negative if end < start.
    """
    delta = end - start
    return int(delta.total_seconds() // 3600)


def iso_to_datetime(iso_str: str) -> datetime:
    """Parse an ISO-8601 string to a datetime object.

    Args:
        iso_str: ISO-8601 formatted datetime string (e.g. '2026-06-15T14:00:00Z').

    Returns:
        Parsed datetime; naive if no timezone info is present.

    Raises:
        ValueError: If the string cannot be parsed as an ISO-8601 datetime.
    """
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot parse ISO-8601 datetime: {iso_str!r}") from exc


def generate_hourly_timestamps(start: datetime, n_hours: int) -> list[datetime]:
    """Generate a sequence of *n_hours* hourly datetime objects starting from *start*.

    Args:
        start: Starting datetime.
        n_hours: Number of hourly timestamps to generate (must be >= 1).

    Returns:
        List of *n_hours* datetime objects spaced one hour apart.

    Raises:
        ValueError: If *n_hours* is less than 1.
    """
    if n_hours < 1:
        raise ValueError(f"n_hours must be at least 1, got {n_hours}")
    return [start + timedelta(hours=i) for i in range(n_hours)]


def is_business_hour(dt: datetime) -> bool:
    """Return True if *dt* falls within standard business hours (Mon-Fri, 08:00-18:00).

    Args:
        dt: Datetime to test (any timezone; local time fields are used).

    Returns:
        True when Monday-Friday and 8:00 <= hour < 18:00.
    """
    return dt.weekday() < 5 and 8 <= dt.hour < 18


def week_of_year(dt: datetime) -> int:
    """Return the ISO week number for *dt* (1-53).

    Args:
        dt: Any datetime.

    Returns:
        Integer ISO week number.
    """
    return dt.isocalendar()[1]
