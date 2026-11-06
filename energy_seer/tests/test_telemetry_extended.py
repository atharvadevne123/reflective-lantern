"""Extended tests for energy_seer app/telemetry.py counters."""

from __future__ import annotations

import pytest

from app.telemetry import (
    all_above,
    counter_exists,
    get,
    increment,
    increment_by,
    reset,
    snapshot,
    top_n,
)


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


class TestSnapshotExtended:
    def setup_method(self) -> None:
        _reset_all()

    def test_empty_snapshot(self) -> None:
        assert snapshot() == {}

    def test_snapshot_contains_all_counters(self) -> None:
        increment("x", 3)
        increment("y", 7)
        s = snapshot()
        assert s["x"] == 3
        assert s["y"] == 7

    def test_snapshot_is_copy_not_reference(self) -> None:
        increment("c", 5)
        s = snapshot()
        increment("c", 10)
        assert s["c"] == 5


class TestAllAboveAndTopNExtended:
    def setup_method(self) -> None:
        _reset_all()

    def test_all_above_empty_counters(self) -> None:
        assert all_above(0) == {}

    def test_all_above_threshold(self) -> None:
        increment("low", 5)
        increment("high", 50)
        result = all_above(10)
        assert "high" in result
        assert "low" not in result

    def test_all_above_zero_includes_all(self) -> None:
        increment("a", 1)
        increment("b", 2)
        result = all_above(0)
        assert "a" in result
        assert "b" in result

    def test_top_n_empty(self) -> None:
        assert top_n(5) == []

    def test_top_n_returns_sorted_descending(self) -> None:
        increment("low", 1)
        increment("mid", 5)
        increment("high", 10)
        result = top_n(3)
        values = [v for _, v in result]
        assert values == sorted(values, reverse=True)

    def test_top_n_limits_results(self) -> None:
        for i in range(10):
            increment(f"c{i}", i)
        result = top_n(3)
        assert len(result) == 3

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_top_n_various_sizes(self, n: int) -> None:
        for i in range(10):
            increment(f"c{i}", i)
        result = top_n(n)
        assert len(result) == min(n, 10)


class TestCounterExistsAndResetExtended:
    def setup_method(self) -> None:
        _reset_all()

    def test_counter_not_exists_before_increment(self) -> None:
        assert not counter_exists("new_counter")

    def test_counter_exists_after_increment(self) -> None:
        increment("c")
        assert counter_exists("c")

    def test_reset_named_counter_to_zero(self) -> None:
        increment("c", 10)
        reset("c")
        assert get("c") == 0

    def test_reset_all_clears_everything(self) -> None:
        increment("a", 5)
        increment("b", 10)
        reset()
        assert get("a") == 0
        assert get("b") == 0

    def test_reset_named_does_not_affect_others(self) -> None:
        increment("keep", 7)
        increment("clear", 3)
        reset("clear")
        assert get("keep") == 7
        assert get("clear") == 0
