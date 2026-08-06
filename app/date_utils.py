"""Date and time utility helpers for Watt-Guard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def utc_now() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=UTC)


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

__all__ = [
    "clamp_to_range",
    "days_until",
    "format_iso",
    "generate_hourly_timestamps",
    "hours_between",
    "is_business_hour",
    "is_weekend",
    "iso_to_datetime",
    "quarter_of_year",
    "round_to_hour",
    "start_of_month",
    "utc_now",
    "week_of_year",
]


def quarter_of_year(dt: datetime) -> int:
    """Return the calendar quarter (1-4) for *dt*.

    Args:
        dt: Any datetime.

    Returns:
        Integer 1, 2, 3, or 4.
    """
    return (dt.month - 1) // 3 + 1


def is_weekend(dt: datetime) -> bool:
    """Return True if *dt* falls on a Saturday or Sunday.

    Args:
        dt: Any datetime.

    Returns:
        True when weekday is 5 (Saturday) or 6 (Sunday).
    """
    return dt.weekday() >= 5


def days_until(dt: datetime, target: datetime) -> int:
    """Return the number of whole days from *dt* to *target*.

    Negative when *target* is in the past relative to *dt*.

    Args:
        dt: Reference datetime.
        target: Target datetime.

    Returns:
        Integer number of days (may be negative).
    """
    return (target.date() - dt.date()).days


def format_iso(dt: datetime) -> str:
    """Format *dt* as an ISO 8601 string with second precision.

    Args:
        dt: Any datetime.

    Returns:
        String like '2024-06-15T14:30:00'.
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def start_of_month(dt: datetime) -> datetime:
    """Return a datetime at midnight on the first day of *dt*'s month.

    Args:
        dt: Any datetime (timezone is preserved).

    Returns:
        Datetime with day=1, hour=0, minute=0, second=0, microsecond=0.
    """
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def clamp_to_range(dt: datetime, start: datetime, end: datetime) -> datetime:
    """Clamp *dt* to the closed interval [start, end].

    Args:
        dt: Datetime to clamp.
        start: Lower bound (inclusive).
        end: Upper bound (inclusive).

    Returns:
        *start* if dt < start, *end* if dt > end, else *dt* unchanged.

    Raises:
        ValueError: If *start* > *end*.
    """
    if start > end:
        raise ValueError(f"start must not be after end: {start} > {end}")
    if dt < start:
        return start
    if dt > end:
        return end
    return dt
