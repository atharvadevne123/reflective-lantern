"""SLA configuration tests for Ticket-Oracle."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.sla_config import SLA_HOURS, sla_deadline, sla_summary, time_to_breach


def test_sla_deadline_p1_gold():
    created = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    deadline = sla_deadline("P1", "Gold", created_at=created)
    delta_minutes = (deadline - created).total_seconds() / 60
    assert delta_minutes == pytest.approx(30.0)  # 1h * 0.5 Gold multiplier


def test_sla_deadline_p4_bronze():
    created = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    deadline = sla_deadline("P4", "Bronze", created_at=created)
    delta_hours = (deadline - created).total_seconds() / 3600
    assert delta_hours == pytest.approx(72.0 * 1.25)


def test_sla_deadline_defaults_to_now():
    before = datetime.now(tz=timezone.utc)
    deadline = sla_deadline("P3", "Standard")
    after = datetime.now(tz=timezone.utc)
    assert before < deadline < after + __import__("datetime").timedelta(hours=25)


def test_time_to_breach_not_yet():
    remaining = time_to_breach("P2", "Standard", ticket_age_minutes=60.0)
    assert remaining > 0


def test_time_to_breach_already_breached():
    remaining = time_to_breach("P1", "Bronze", ticket_age_minutes=120.0)
    assert remaining < 0


def test_time_to_breach_exact():
    remaining = time_to_breach("P1", "Standard", ticket_age_minutes=60.0)
    assert remaining == pytest.approx(0.0)


def test_sla_summary_not_breached():
    summary = sla_summary("P3", "Gold", ticket_age_minutes=100.0)
    assert not summary["breached"]
    assert summary["minutes_remaining"] > 0
    assert summary["priority"] == "P3"


def test_sla_summary_breached():
    summary = sla_summary("P1", "Standard", ticket_age_minutes=200.0)
    assert summary["breached"]


def test_all_priorities_have_sla():
    for p in ["P1", "P2", "P3", "P4"]:
        assert p in SLA_HOURS
        assert SLA_HOURS[p] > 0
