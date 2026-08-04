"""Tests for energy_seer/app/telemetry.py."""

from __future__ import annotations

import pytest


def _reset_all() -> None:
    from energy_seer.app.telemetry import reset

    reset()


class TestIncrement:
    def setup_method(self) -> None:
        _reset_all()

    def test_basic_increment(self) -> None:
        from energy_seer.app.telemetry import get, increment

        increment("req_count")
        assert get("req_count") == 1

    def test_multiple_increments(self) -> None:
        from energy_seer.app.telemetry import get, increment

        increment("req_count", 5)
        assert get("req_count") == 5

    def test_new_counter_starts_at_zero(self) -> None:
        from energy_seer.app.telemetry import get

        assert get("nonexistent") == 0


class TestIncrementBy:
    def setup_method(self) -> None:
        _reset_all()

    def test_positive_amount(self) -> None:
        from energy_seer.app.telemetry import get, increment_by

        increment_by("x", 10)
        assert get("x") == 10

    def test_zero_amount_ok(self) -> None:
        from energy_seer.app.telemetry import get, increment_by

        increment_by("x", 0)
        assert get("x") == 0

    def test_negative_raises(self) -> None:
        from energy_seer.app.telemetry import increment_by

        with pytest.raises(ValueError):
            increment_by("x", -1)


class TestSnapshot:
    def setup_method(self) -> None:
        _reset_all()

    def test_returns_copy(self) -> None:
        from energy_seer.app.telemetry import increment, snapshot

        increment("a")
        s = snapshot()
        assert "a" in s
        assert isinstance(s, dict)


class TestAllAbove:
    def setup_method(self) -> None:
        _reset_all()

    def test_filters_below(self) -> None:
        from energy_seer.app.telemetry import all_above, increment

        increment("a", 5)
        increment("b", 1)
        result = all_above(3)
        assert "a" in result
        assert "b" not in result

    def test_empty_when_none_above(self) -> None:
        from energy_seer.app.telemetry import all_above

        assert all_above(999) == {}


class TestTopN:
    def setup_method(self) -> None:
        _reset_all()

    def test_returns_sorted(self) -> None:
        from energy_seer.app.telemetry import increment, top_n

        increment("low", 1)
        increment("high", 10)
        result = top_n(2)
        assert result[0][0] == "high"
        assert result[1][0] == "low"

    def test_limits_to_n(self) -> None:
        from energy_seer.app.telemetry import increment, top_n

        for i in range(10):
            increment(f"counter_{i}", i)
        assert len(top_n(3)) == 3


class TestReset:
    def setup_method(self) -> None:
        _reset_all()

    def test_reset_specific(self) -> None:
        from energy_seer.app.telemetry import get, increment, reset

        increment("x", 5)
        reset("x")
        assert get("x") == 0

    def test_reset_all(self) -> None:
        from energy_seer.app.telemetry import get, increment, reset

        increment("a", 3)
        increment("b", 7)
        reset()
        assert get("a") == 0
        assert get("b") == 0
