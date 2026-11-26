"""Extended SLA configuration tests — second pass with parametrize and boundary coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.sla_config import (
    SLA_HOURS,
    TIER_MULTIPLIERS,
    sla_deadline,
    sla_summary,
    time_to_breach,
)


def test_sla_deadline_p1_gold_fastest() -> None:
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    deadline = sla_deadline("P1", "Gold", created_at=t0)
    expected = t0 + timedelta(hours=0.5)
    assert deadline == expected


def test_sla_deadline_p4_standard_72h() -> None:
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P4", "Standard", created_at=t0)
    expected = t0 + timedelta(hours=72.0)
    assert deadline == expected


def test_sla_deadline_unknown_priority_defaults_24h() -> None:
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P5", "Standard", created_at=t0)
    expected = t0 + timedelta(hours=24.0)
    assert deadline == expected


def test_sla_deadline_unknown_tier_defaults_multiplier_one() -> None:
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P2", "Platinum", created_at=t0)
    expected = t0 + timedelta(hours=4.0)
    assert deadline == expected


def test_sla_deadline_defaults_to_now_when_no_created_at() -> None:
    before = datetime.now(tz=UTC)
    deadline = sla_deadline("P3", "Standard")
    after = datetime.now(tz=UTC)
    assert before < deadline < after + timedelta(hours=25)


@pytest.mark.parametrize(
    "priority,tier,expected_hours",
    [
        ("P1", "Gold", 0.5),
        ("P1", "Silver", 0.75),
        ("P1", "Standard", 1.0),
        ("P2", "Gold", 2.0),
        ("P2", "Standard", 4.0),
        ("P3", "Standard", 24.0),
        ("P4", "Standard", 72.0),
        ("P4", "Bronze", 90.0),
    ],
)
def test_time_to_breach_zero_age_equals_window(
    priority: str, tier: str, expected_hours: float
) -> None:
    remaining = time_to_breach(priority, tier, ticket_age_minutes=0.0)
    assert remaining == pytest.approx(expected_hours * 60.0)


def test_time_to_breach_already_breached_negative() -> None:
    remaining = time_to_breach("P1", "Standard", ticket_age_minutes=120.0)
    assert remaining < 0


def test_time_to_breach_just_at_boundary() -> None:
    sla_minutes = SLA_HOURS["P1"] * TIER_MULTIPLIERS["Standard"] * 60.0
    remaining = time_to_breach("P1", "Standard", ticket_age_minutes=sla_minutes)
    assert remaining == pytest.approx(0.0, abs=0.01)


def test_sla_summary_has_required_keys() -> None:
    summary = sla_summary("P2", "Gold", ticket_age_minutes=60.0)
    for key in ("priority", "customer_tier", "sla_window_hours", "minutes_remaining", "breached"):
        assert key in summary


def test_sla_summary_breached_false_when_young() -> None:
    summary = sla_summary("P4", "Standard", ticket_age_minutes=1.0)
    assert summary["breached"] is False


def test_sla_summary_breached_true_when_old() -> None:
    summary = sla_summary("P1", "Standard", ticket_age_minutes=200.0)
    assert summary["breached"] is True


def test_sla_summary_priority_and_tier_stored() -> None:
    summary = sla_summary("P3", "Silver", ticket_age_minutes=0.0)
    assert summary["priority"] == "P3"
    assert summary["customer_tier"] == "Silver"


def test_sla_hours_all_priorities_defined() -> None:
    for p in ("P1", "P2", "P3", "P4"):
        assert p in SLA_HOURS
        assert SLA_HOURS[p] > 0


def test_tier_multipliers_all_tiers_defined() -> None:
    for tier in ("Gold", "Silver", "Standard", "Bronze"):
        assert tier in TIER_MULTIPLIERS
        assert TIER_MULTIPLIERS[tier] > 0
