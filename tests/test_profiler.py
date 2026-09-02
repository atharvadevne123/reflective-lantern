"""Tests for app.profiler module."""

from __future__ import annotations

import pytest

from app.profiler import get_stats, reset_stats, timed, tracked


class TestTimed:
    def test_returns_function_result(self):
        @timed()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self):
        @timed()
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_custom_label_accepted(self):
        @timed(label="custom")
        def fn():
            return 42

        assert fn() == 42

    def test_propagates_exception(self):
        @timed()
        def bad():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()


class TestTracked:
    def setup_method(self):
        reset_stats()

    def test_stats_recorded_after_call(self):
        @tracked(label="test_fn")
        def fn():
            return 1

        fn()
        stats = get_stats("test_fn")
        assert stats["calls"] == 1
        assert stats["total_ms"] >= 0

    def test_multiple_calls_accumulate(self):
        @tracked(label="multi")
        def fn():
            pass

        fn()
        fn()
        fn()
        assert get_stats("multi")["calls"] == 3

    def test_min_max_updated(self):
        @tracked(label="minmax")
        def fn():
            pass

        fn()
        fn()
        stats = get_stats("minmax")
        assert stats["min_ms"] <= stats["max_ms"]

    def test_avg_ms_computed(self):
        @tracked(label="avg")
        def fn():
            pass

        fn()
        fn()
        stats = get_stats("avg")
        # to_dict() rounds total_ms and avg_ms independently to 3dp, so exact
        # equality is not guaranteed: a total of 0.003 reports avg 0.001 while
        # total/2 is 0.0015. Allow one rounding step of slack.
        assert stats["avg_ms"] == pytest.approx(stats["total_ms"] / 2, abs=0.001)

    def test_get_all_stats(self):
        @tracked(label="a_func")
        def a():
            pass

        a()
        all_stats = get_stats()
        assert "a_func" in all_stats

    def test_unknown_label_returns_empty(self):
        reset_stats()
        assert get_stats("nonexistent") == {}

    def test_reset_specific_label(self):
        @tracked(label="reset_me")
        def fn():
            pass

        fn()
        reset_stats("reset_me")
        assert get_stats("reset_me")["calls"] == 0

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_call_count_matches(self, n):
        @tracked(label=f"count_{n}")
        def fn():
            pass

        for _ in range(n):
            fn()
        assert get_stats(f"count_{n}")["calls"] == n

    def test_total_time_non_negative(self) -> None:
        reset_stats()

        @tracked(label="timing_check")
        def fn():
            pass

        fn()
        assert get_stats("timing_check")["total_time"] >= 0.0

    def test_stats_have_required_keys(self) -> None:
        reset_stats()

        @tracked(label="keys_check")
        def fn():
            pass

        fn()
        stats = get_stats("keys_check")
        assert "calls" in stats and "total_time" in stats
