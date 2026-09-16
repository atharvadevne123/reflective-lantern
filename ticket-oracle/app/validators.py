"""Input validation utilities for Ticket-Oracle.

Provides standalone validation helpers used by the API layer to enforce
business-rule constraints beyond what Pydantic schema validators cover.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TICKET_ID_RE = re.compile(r"^[A-Za-z0-9\-_]{1,64}$")
_MAX_DESCRIPTION_WORDS = 400


def validate_ticket_id(ticket_id: str) -> str:
    """Validate that ticket_id matches the allowed pattern.

    Args:
        ticket_id: Raw ticket identifier string.

    Returns:
        Stripped and validated ticket ID.

    Raises:
        ValueError: If the ID contains disallowed characters or is empty.
    """
    tid = ticket_id.strip()
    if not tid:
        raise ValueError("ticket_id must not be empty")
    if not _TICKET_ID_RE.match(tid):
        raise ValueError(f"ticket_id '{tid}' contains invalid characters (allowed: A-Z, a-z, 0-9, -, _)")
    return tid


def validate_description(description: str) -> str:
    """Validate ticket description length and content.

    Args:
        description: Raw description text.

    Returns:
        Stripped description.

    Raises:
        ValueError: If description is too short, too long, or empty.
    """
    desc = description.strip()
    if len(desc) < 3:
        raise ValueError("description must be at least 3 characters")
    if len(desc) > 2000:
        raise ValueError("description exceeds 2000 character limit")
    word_count = len(desc.split())
    if word_count > _MAX_DESCRIPTION_WORDS:
        logger.warning("description_truncated", extra={"word_count": word_count})
    return desc


def validate_hour_of_day(hour: int) -> int:
    """Validate hour is in [0, 23].

    Args:
        hour: Hour value to validate.

    Returns:
        Validated hour.

    Raises:
        ValueError: If hour is outside the valid range.
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour_of_day must be in [0, 23], got {hour}")
    return hour


def validate_day_of_week(day: int) -> int:
    """Validate day of week is in [0, 6].

    Args:
        day: Day of week value (0=Monday, 6=Sunday).

    Returns:
        Validated day.

    Raises:
        ValueError: If day is outside the valid range.
    """
    if not 0 <= day <= 6:
        raise ValueError(f"day_of_week must be in [0, 6], got {day}")
    return day


def validate_workload_fields(open_tickets: int, agent_load: int) -> tuple[int, int]:
    """Validate workload counters are non-negative.

    Args:
        open_tickets: Current open ticket queue count.
        agent_load: Agent's current ticket assignment count.

    Returns:
        Tuple of (open_tickets, agent_load).

    Raises:
        ValueError: If either value is negative.
    """
    if open_tickets < 0:
        raise ValueError(f"open_tickets_count must be >= 0, got {open_tickets}")
    if agent_load < 0:
        raise ValueError(f"agent_load must be >= 0, got {agent_load}")
    return open_tickets, agent_load


def sanitise_text_input(text: str, max_len: int = 2000) -> str:
    """Strip whitespace and truncate text to max_len for safety.

    Args:
        text: Raw text input.
        max_len: Maximum allowed character length.

    Returns:
        Sanitised text.
    """
    return text.strip()[:max_len]
