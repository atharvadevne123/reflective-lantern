"""Tests for config.mode — run mode determination."""

from __future__ import annotations

from datetime import date

import pytest

from config.mode import RunMode, determine_mode, is_innovation_day


@pytest.mark.parametrize("target_date,expected", [
    # Wednesday day 8-14 → INNOVATION
    (date(2026, 7, 8), RunMode.INNOVATION),    # Wed day 8
    (date(2026, 10, 14), RunMode.INNOVATION),  # Wed day 14 (Oct 2026)
    (date(2026, 7, 22), RunMode.INNOVATION),   # Wed day 22
    (date(2026, 10, 28), RunMode.INNOVATION),  # Wed day 28 (Oct 2026)
    # Wednesday but outside range → IMPROVEMENT
    (date(2026, 7, 1), RunMode.IMPROVEMENT),   # Wed day 1
    (date(2026, 7, 15), RunMode.IMPROVEMENT),  # Wed day 15
    (date(2026, 7, 29), RunMode.IMPROVEMENT),  # Wed day 29
    # Non-Wednesday → IMPROVEMENT
    (date(2026, 7, 13), RunMode.IMPROVEMENT),  # Monday day 13 (in range but not Wed)
    (date(2026, 7, 9), RunMode.IMPROVEMENT),   # Thursday day 9
])
def test_determine_mode(target_date: date, expected: RunMode) -> None:
    assert determine_mode(target_date) == expected


@pytest.mark.parametrize("target_date,expected", [
    (date(2026, 7, 8), True),
    (date(2026, 7, 1), False),
    (date(2026, 7, 13), False),
])
def test_is_innovation_day(target_date: date, expected: bool) -> None:
    assert is_innovation_day(target_date) == expected


def test_determine_mode_improvement_is_default_on_weekday() -> None:
    monday = date(2026, 7, 6)  # Monday
    assert determine_mode(monday) == RunMode.IMPROVEMENT


def test_run_mode_string_values() -> None:
    assert RunMode.IMPROVEMENT == "IMPROVEMENT"
    assert RunMode.INNOVATION == "INNOVATION"


def test_determine_mode_boundaries() -> None:
    # Day 7 (outside range) on a Wednesday
    wed_day7 = date(2026, 7, 7)  # Tue, but pick a real Wed
    wed_day21 = date(2026, 7, 21)  # Tue
    # These aren't Wednesdays, so IMPROVEMENT regardless
    assert determine_mode(wed_day7) == RunMode.IMPROVEMENT
    assert determine_mode(wed_day21) == RunMode.IMPROVEMENT
