"""Date and time utility helpers for Watt-Guard."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


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
        logger.debug("iso_to_datetime: parse failed for %r", iso_str)
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
    "datetime_to_iso",
    "days_between",
    "generate_hourly_timestamps",
    "hours_between",
    "is_business_hour",
    "iso_to_datetime",
    "round_to_hour",
    "start_of_day",
    "utc_now",
    "week_of_year",
]


def datetime_to_iso(dt: datetime) -> str:
    """Serialise *dt* to an ISO-8601 string with UTC offset.

    Args:
        dt: Datetime to serialise (naive datetimes are assumed UTC).

    Returns:
        ISO-8601 string, e.g. '2026-08-03T12:00:00+00:00'.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def start_of_day(dt: datetime) -> datetime:
    """Return midnight (00:00:00) of the same day as *dt*.

    Args:
        dt: Any datetime.

    Returns:
        Datetime with time zeroed to midnight, preserving tzinfo.
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def days_between(start: datetime, end: datetime) -> int:
    """Return the number of whole calendar days between *start* and *end*.

    Args:
        start: Earlier datetime.
        end: Later datetime.

    Returns:
        Integer day count (can be negative if end < start).
    """
    delta = end - start
    return delta.days


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds (non-negative).

    Returns:
        Human-readable string such as '2h 5m 3s', '45m 0s', or '12s'.

    Raises:
        ValueError: If *seconds* is negative.
    """
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def is_leap_year(year: int) -> bool:
    """Return True if *year* is a leap year.

    Args:
        year: Calendar year (positive integer).

    Returns:
        True when the year has 366 days, False otherwise.

    Raises:
        ValueError: If *year* is not a positive integer.
    """
    if year <= 0:
        raise ValueError(f"year must be positive, got {year}")
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    """Return the number of days in *month* of *year*.

    Args:
        year: Calendar year.
        month: Month number 1-12.

    Returns:
        Number of days in the given month.

    Raises:
        ValueError: If *month* is not in 1-12.
    """
    if month < 1 or month > 12:
        raise ValueError(f"month must be 1-12, got {month}")
    import calendar
    return calendar.monthrange(year, month)[1]


def next_business_day(dt: datetime) -> datetime:
    """Return the next business day (Monday-Friday) after *dt*.

    If *dt* falls on a weekday, returns the next weekday.
    If *dt* falls on Saturday, returns Monday.
    If *dt* falls on Sunday, returns Monday.

    Args:
        dt: Reference datetime.

    Returns:
        Datetime at midnight of the next business day, same tzinfo as *dt*.
    """
    import datetime as _dt
    d = dt.date() + _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return datetime.combine(d, datetime.min.time()).replace(tzinfo=dt.tzinfo)
