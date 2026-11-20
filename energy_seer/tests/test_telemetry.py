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


class TestSnapshotAndBulk:
    def setup_method(self) -> None:
        from app.telemetry import reset

        reset()

    def test_snapshot_returns_copy(self) -> None:
        from app.telemetry import increment, snapshot

        increment("a", 3)
        snap = snapshot()
        assert snap.get("a") == 3

    def test_snapshot_is_isolated(self) -> None:
        from app.telemetry import increment, snapshot

        increment("b", 1)
        snap = snapshot()
        increment("b", 99)
        # Snapshot taken before second increment should not reflect it
        assert snap.get("b") == 1

    def test_increment_accumulates_across_calls(self) -> None:
        from app.telemetry import get, increment

        for _ in range(5):
            increment("c")
        assert get("c") == 5


class TestCounterExists:
    def test_exists_after_increment(self) -> None:
        from app.telemetry import counter_exists, increment, reset

        reset("ce_test")
        increment("ce_test")
        assert counter_exists("ce_test") is True

    def test_new_counter_not_exists(self) -> None:
        from app.telemetry import counter_exists, reset

        reset("ce_new_xyz")
        assert counter_exists("ce_new_xyz") is False

    def test_reset_does_not_delete_counter(self) -> None:
        from app.telemetry import counter_exists, increment, reset

        increment("ce_check")
        reset("ce_check")
        # reset sets to 0 but keeps the key
        assert counter_exists("ce_check") is True


class TestResetPrefix:
    def test_resets_matching_counters(self) -> None:
        from app.telemetry import get, increment, reset_prefix

        increment("api_get")
        increment("api_post")
        reset_prefix("api_")
        assert get("api_get") == 0
        assert get("api_post") == 0

    def test_returns_count_of_reset(self) -> None:
        from app.telemetry import increment, reset_prefix

        increment("pfx_a")
        increment("pfx_b")
        count = reset_prefix("pfx_")
        assert count >= 2
