"""Time and datetime utility functions for scheduling and duration math."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

__all__ = [
    "format_duration_seconds",
    "is_business_hours",
    "iso_now",
    "parse_iso",
    "seconds_until",
    "utc_now",
]


def utc_now() -> datetime.datetime:
    """Return the current UTC time as a timezone-aware datetime.

    Returns:
        Current UTC datetime with tzinfo set to UTC.
    """
    return datetime.datetime.now(tz=datetime.timezone.utc)


def iso_now() -> str:
    """Return the current UTC time formatted as an ISO-8601 string.

    Returns:
        ISO-8601 formatted UTC timestamp, e.g. ``"2024-01-15T10:30:00+00:00"``.
    """
    return utc_now().isoformat()


def parse_iso(timestamp: str) -> datetime.datetime:
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime.

    Args:
        timestamp: ISO-8601 string such as ``"2024-01-15T10:30:00+00:00"`` or
            ``"2024-01-15T10:30:00Z"``.

    Returns:
        Timezone-aware datetime. UTC is assumed when no timezone is given.
    """
    ts = timestamp.rstrip("Z")
    dt = datetime.datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def seconds_until(target: datetime.datetime) -> float:
    """Return the number of seconds from now until *target*.

    Args:
        target: A timezone-aware future datetime.

    Returns:
        Positive float when *target* is in the future; negative when in the past.
    """
    now = datetime.datetime.now(tz=target.tzinfo or datetime.timezone.utc)
    return (target - now).total_seconds()


def format_duration_seconds(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Examples:
        - 45 -> ``"45s"``
        - 90 -> ``"1m 30s"``
        - 3661 -> ``"1h 1m 1s"``

    Args:
        seconds: Duration in seconds (non-negative float).

    Returns:
        Human-readable duration string.
    """
    total = int(abs(seconds))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    parts: list[str] = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def is_business_hours(
    dt: datetime.datetime,
    start_hour: int = 9,
    end_hour: int = 17,
    tz_name: str = "UTC",
) -> bool:
    """Return True if *dt* falls within business hours on a weekday.

    Args:
        dt: The datetime to check. Converted to *tz_name* before evaluation.
        start_hour: Start of business hours (inclusive). Defaults to 9.
        end_hour: End of business hours (exclusive). Defaults to 17.
        tz_name: IANA timezone name for business-hours calculation. Defaults to ``"UTC"``.

    Returns:
        True when *dt* is a weekday and within [start_hour, end_hour).
    """
    tz = ZoneInfo(tz_name)
    local = dt.astimezone(tz)
    if local.weekday() >= 5:
        return False
    return start_hour <= local.hour < end_hour
