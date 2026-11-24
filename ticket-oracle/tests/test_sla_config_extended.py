"""Extended SLA configuration tests with parametrize and boundary coverage."""
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


@pytest.mark.parametrize("priority,hours", [
    ("P1", 1.0),
    ("P2", 4.0),
    ("P3", 24.0),
    ("P4", 72.0),
])
def test_sla_deadline_base_hours(priority: str, hours: float) -> None:
    created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    deadline = sla_deadline(priority, "Standard", created_at=created)
    expected = created + timedelta(hours=hours)
    assert deadline == expected


@pytest.mark.parametrize("tier,multiplier", [
    ("Gold", 0.5),
    ("Silver", 0.75),
    ("Standard", 1.0),
    ("Bronze", 1.25),
])
def test_sla_deadline_tier_multipliers(tier: str, multiplier: float) -> None:
    created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P3", tier, created_at=created)
    expected = created + timedelta(hours=24.0 * multiplier)
    assert abs((deadline - expected).total_seconds()) < 1


def test_sla_deadline_unknown_priority_defaults_to_24h() -> None:
    created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P9", "Standard", created_at=created)
    expected = created + timedelta(hours=24.0)
    assert deadline == expected


def test_sla_deadline_unknown_tier_defaults_to_1x() -> None:
    created = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    deadline = sla_deadline("P3", "Platinum", created_at=created)
    expected = created + timedelta(hours=24.0)
    assert deadline == expected


def test_sla_deadline_naive_datetime_treated_as_utc() -> None:
    created = datetime(2024, 1, 1, 12, 0, 0)
    deadline = sla_deadline("P1", "Standard", created_at=created)
    assert deadline.tzinfo is not None


def test_sla_deadline_defaults_to_now_when_no_created_at() -> None:
    before = datetime.now(tz=UTC)
    deadline = sla_deadline("P1", "Standard")
    after = datetime.now(tz=UTC)
    expected_min = before + timedelta(hours=1.0 * 1.0)
    expected_max = after + timedelta(hours=1.0 * 1.0)
    assert expected_min <= deadline <= expected_max


@pytest.mark.parametrize("priority,tier,age_min,expected_sign", [
    ("P3", "Standard", 60.0, 1),
    ("P3", "Standard", 1440.0, 1),
    ("P3", "Standard", 1441.0, -1),
    ("P1", "Gold", 29.0, 1),
    ("P1", "Gold", 31.0, -1),
])
def test_time_to_breach_sign(
    priority: str, tier: str, age_min: float, expected_sign: int
) -> None:
    remaining = time_to_breach(priority, tier, age_min)
    assert (remaining > 0) == (expected_sign > 0)


def test_time_to_breach_zero_age() -> None:
    remaining = time_to_breach("P2", "Standard", 0.0)
    assert remaining == pytest.approx(4.0 * 60.0, rel=1e-6)


def test_time_to_breach_exact_breach_zero() -> None:
    window_minutes = SLA_HOURS["P3"] * TIER_MULTIPLIERS["Standard"] * 60.0
    remaining = time_to_breach("P3", "Standard", window_minutes)
    assert remaining == pytest.approx(0.0, abs=1e-6)


def test_sla_summary_has_required_keys() -> None:
    result = sla_summary("P2", "Gold", 60.0)
    for key in ("priority", "customer_tier", "sla_window_hours", "minutes_remaining", "breached"):
        assert key in result


def test_sla_summary_breached_false_when_within_window() -> None:
    result = sla_summary("P4", "Standard", 100.0)
    assert result["breached"] is False


def test_sla_summary_breached_true_when_over_window() -> None:
    result = sla_summary("P1", "Standard", 70.0)
    assert result["breached"] is True


@pytest.mark.parametrize("priority,tier", [
    ("P1", "Gold"),
    ("P2", "Silver"),
    ("P3", "Standard"),
    ("P4", "Bronze"),
])
def test_sla_summary_window_hours_matches_calculation(priority: str, tier: str) -> None:
    result = sla_summary(priority, tier, 0.0)
    expected = SLA_HOURS[priority] * TIER_MULTIPLIERS[tier]
    assert result["sla_window_hours"] == pytest.approx(expected, rel=1e-6)


def test_sla_summary_breached_is_bool() -> None:
    result = sla_summary("P2", "Standard", 300.0)
    assert isinstance(result["breached"], bool)
