"""Validation utility tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.validators import (
    sanitise_text_input,
    validate_day_of_week,
    validate_description,
    validate_hour_of_day,
    validate_ticket_id,
    validate_workload_fields,
)


def test_validate_ticket_id_valid():
    assert validate_ticket_id("T-12345") == "T-12345"


def test_validate_ticket_id_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_ticket_id("")


def test_validate_ticket_id_invalid_chars():
    with pytest.raises(ValueError, match="invalid characters"):
        validate_ticket_id("T/123 !")


def test_validate_ticket_id_strips_whitespace():
    assert validate_ticket_id("  T-001  ") == "T-001"


def test_validate_description_valid():
    assert validate_description("VPN not working") == "VPN not working"


def test_validate_description_too_short():
    with pytest.raises(ValueError, match="at least 3 characters"):
        validate_description("ab")


def test_validate_description_too_long():
    with pytest.raises(ValueError, match="2000 character"):
        validate_description("x" * 2001)


def test_validate_description_strips_whitespace():
    result = validate_description("  Hello world  ")
    assert result == "Hello world"


def test_validate_hour_valid():
    assert validate_hour_of_day(0) == 0
    assert validate_hour_of_day(23) == 23
    assert validate_hour_of_day(12) == 12


def test_validate_hour_negative_raises():
    with pytest.raises(ValueError):
        validate_hour_of_day(-1)


def test_validate_hour_too_large_raises():
    with pytest.raises(ValueError):
        validate_hour_of_day(24)


def test_validate_day_of_week_valid():
    assert validate_day_of_week(0) == 0
    assert validate_day_of_week(6) == 6


def test_validate_day_of_week_invalid():
    with pytest.raises(ValueError):
        validate_day_of_week(7)


def test_validate_workload_valid():
    ot, al = validate_workload_fields(10, 5)
    assert ot == 10
    assert al == 5


def test_validate_workload_negative_open_tickets():
    with pytest.raises(ValueError, match="open_tickets_count"):
        validate_workload_fields(-1, 5)


def test_validate_workload_negative_agent_load():
    with pytest.raises(ValueError, match="agent_load"):
        validate_workload_fields(10, -1)


def test_sanitise_text_truncates():
    result = sanitise_text_input("a" * 3000, max_len=100)
    assert len(result) == 100


def test_sanitise_text_strips():
    assert sanitise_text_input("  hello  ") == "hello"
