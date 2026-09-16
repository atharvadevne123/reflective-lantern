"""Schema validation tests for Ticket-Oracle."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BatchTicketPayload, TicketPayload


VALID_PAYLOAD = {
    "ticket_id": "T-001",
    "description": "Server is down and users cannot log in to the system",
    "department": "IT",
    "incident_type": "outage",
    "channel": "email",
    "customer_tier": "Gold",
    "product_area": "auth",
    "open_tickets_count": 5,
    "agent_load": 0.6,
    "hour_of_day": 14,
    "day_of_week": 2,
    "ticket_age_minutes": 30.0,
    "same_user_last_7d": 2,
}


def test_valid_payload_parses():
    p = TicketPayload(**VALID_PAYLOAD)
    assert p.ticket_id == "T-001"
    assert p.priority_field_absent_ok := True  # noqa: F841


def test_valid_payload_parses():  # noqa: F811
    p = TicketPayload(**VALID_PAYLOAD)
    assert p.channel == "email"


def test_invalid_channel_raises():
    bad = {**VALID_PAYLOAD, "channel": "fax"}
    with pytest.raises(ValidationError, match="channel"):
        TicketPayload(**bad)


def test_invalid_tier_raises():
    bad = {**VALID_PAYLOAD, "customer_tier": "Platinum"}
    with pytest.raises(ValidationError, match="customer_tier"):
        TicketPayload(**bad)


def test_all_valid_channels():
    for ch in ["email", "phone", "chat", "portal"]:
        p = TicketPayload(**{**VALID_PAYLOAD, "channel": ch})
        assert p.channel == ch


def test_all_valid_tiers():
    for tier in ["Bronze", "Standard", "Silver", "Gold"]:
        p = TicketPayload(**{**VALID_PAYLOAD, "customer_tier": tier})
        assert p.customer_tier == tier


def test_hour_of_day_bounds():
    with pytest.raises(ValidationError):
        TicketPayload(**{**VALID_PAYLOAD, "hour_of_day": 24})
    with pytest.raises(ValidationError):
        TicketPayload(**{**VALID_PAYLOAD, "hour_of_day": -1})


def test_agent_load_bounds():
    with pytest.raises(ValidationError):
        TicketPayload(**{**VALID_PAYLOAD, "agent_load": 1.5})


def test_description_too_short():
    with pytest.raises(ValidationError):
        TicketPayload(**{**VALID_PAYLOAD, "description": "short"})


def test_batch_payload_single():
    b = BatchTicketPayload(tickets=[VALID_PAYLOAD])
    assert len(b.tickets) == 1


def test_batch_payload_empty_raises():
    with pytest.raises(ValidationError):
        BatchTicketPayload(tickets=[])


def test_batch_payload_multi():
    t2 = {**VALID_PAYLOAD, "ticket_id": "T-002"}
    b = BatchTicketPayload(tickets=[VALID_PAYLOAD, t2])
    assert len(b.tickets) == 2
