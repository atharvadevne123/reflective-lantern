"""SLA configuration and deadline calculation for Ticket-Oracle.

Defines per-priority SLA windows and helpers for computing breach
deadlines and remaining time given ticket metadata.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

SLA_HOURS: dict[str, float] = {
    "P1": 1.0,
    "P2": 4.0,
    "P3": 24.0,
    "P4": 72.0,
}

TIER_MULTIPLIERS: dict[str, float] = {
    "Gold": 0.5,
    "Silver": 0.75,
    "Standard": 1.0,
    "Bronze": 1.25,
}


def sla_deadline(
    priority: str,
    customer_tier: str,
    created_at: datetime | None = None,
) -> datetime:
    """Compute the SLA deadline for a ticket.

    Args:
        priority: Ticket priority (P1-P4).
        customer_tier: Customer tier affecting SLA multiplier.
        created_at: Ticket creation time (UTC); defaults to now.

    Returns:
        UTC datetime of the SLA deadline.
    """
    base_hours = SLA_HOURS.get(priority, 24.0)
    multiplier = TIER_MULTIPLIERS.get(customer_tier, 1.0)
    window = timedelta(hours=base_hours * multiplier)
    origin = created_at or datetime.now(tz=timezone.utc)
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=timezone.utc)
    return origin + window


def time_to_breach(
    priority: str,
    customer_tier: str,
    ticket_age_minutes: float,
) -> float:
    """Return remaining minutes before SLA breach (negative = already breached).

    Args:
        priority: Ticket priority.
        customer_tier: Customer tier.
        ticket_age_minutes: How many minutes since the ticket was opened.

    Returns:
        Float minutes remaining (negative when already breached).
    """
    base_hours = SLA_HOURS.get(priority, 24.0)
    multiplier = TIER_MULTIPLIERS.get(customer_tier, 1.0)
    window_minutes = base_hours * multiplier * 60.0
    return round(window_minutes - ticket_age_minutes, 2)


def sla_summary(
    priority: str,
    customer_tier: str,
    ticket_age_minutes: float,
) -> dict[str, Any]:
    """Return a structured SLA summary dict."""
    remaining = time_to_breach(priority, customer_tier, ticket_age_minutes)
    return {
        "priority": priority,
        "customer_tier": customer_tier,
        "sla_window_hours": SLA_HOURS.get(priority, 24.0) * TIER_MULTIPLIERS.get(customer_tier, 1.0),
        "minutes_remaining": remaining,
        "breached": remaining < 0,
    }
