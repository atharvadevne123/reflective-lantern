"""Extended tests for energy_seer app/telemetry.py counters."""

from __future__ import annotations

import pytest

from app.telemetry import get, increment, increment_by, reset


def _reset_all() -> None:
    reset()


class TestIncrementExtended:
    def setup_method(self) -> None:
        _reset_all()

    def test_increment_default_amount_is_one(self) -> None:
        increment("c")
        assert get("c") == 1

    def test_increment_multiple_times_accumulates(self) -> None:
        for _ in range(5):
            increment("c")
        assert get("c") == 5

    def test_increment_negative_amount_decrements(self) -> None:
        increment("c", 10)
        increment("c", -3)
        assert get("c") == 7

    def test_different_counters_independent(self) -> None:
        increment("a", 5)
        increment("b", 10)
        assert get("a") == 5
        assert get("b") == 10

    def test_missing_counter_returns_zero(self) -> None:
        assert get("never_touched") == 0

    @pytest.mark.parametrize("amount", [1, 5, 100, 1000])
    def test_increment_by_amount(self, amount: int) -> None:
        increment("c", amount)
        assert get("c") == amount


class TestIncrementByExtended:
    def setup_method(self) -> None:
        _reset_all()

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            increment_by("c", -1)

    def test_zero_amount_accepted(self) -> None:
        increment_by("c", 0)
        assert get("c") == 0

    def test_large_amount(self) -> None:
        increment_by("c", 1_000_000)
        assert get("c") == 1_000_000

    def test_multiple_increment_by_accumulate(self) -> None:
        increment_by("c", 10)
        increment_by("c", 20)
        assert get("c") == 30
