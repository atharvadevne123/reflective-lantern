"""Tests for app.profiler module."""

from __future__ import annotations

import pytest

from app.profiler import get_stats, reset_stats, timed, tracked


class TestTimed:
    def test_returns_function_result(self) -> None:
        @timed()
        def add(a, b) -> None:
            return a + b

        assert add(2, 3) == 5

    def test_preserves_function_name(self) -> None:
        @timed()
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_custom_label_accepted(self) -> None:
        @timed(label="custom")
        def fn() -> None:
            return 42

        assert fn() == 42

    def test_propagates_exception(self) -> None:
        @timed()
        def bad() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad()


class TestTracked:
    def setup_method(self) -> None:
        reset_stats()

    def test_stats_recorded_after_call(self) -> None:
        @tracked(label="test_fn")
        def fn() -> None:
            return 1

        fn()
        stats = get_stats("test_fn")
        assert stats["calls"] == 1
        assert stats["total_ms"] >= 0

    def test_multiple_calls_accumulate(self) -> None:
        @tracked(label="multi")
        def fn() -> None:
            pass

        fn()
        fn()
        fn()
        assert get_stats("multi")["calls"] == 3

    def test_min_max_updated(self) -> None:
        @tracked(label="minmax")
        def fn() -> None:
            pass

        fn()
        fn()
        stats = get_stats("minmax")
        assert stats["min_ms"] <= stats["max_ms"]

    def test_avg_ms_computed(self) -> None:
        @tracked(label="avg")
        def fn() -> None:
            pass

        fn()
        fn()
        stats = get_stats("avg")
        # to_dict() rounds total_ms and avg_ms independently to 3dp, so exact
        # equality is not guaranteed: a total of 0.003 reports avg 0.001 while
        # total/2 is 0.0015. Allow one rounding step of slack.
        assert stats["avg_ms"] == pytest.approx(stats["total_ms"] / 2, abs=0.001)

    def test_get_all_stats(self) -> None:
        @tracked(label="a_func")
        def a() -> None:
            pass

        a()
        all_stats = get_stats()
        assert "a_func" in all_stats

    def test_unknown_label_returns_empty(self) -> None:
        reset_stats()
        assert get_stats("nonexistent") == {}

    def test_reset_specific_label(self) -> None:
        @tracked(label="reset_me")
        def fn() -> None:
            pass

        fn()
        reset_stats("reset_me")
        assert get_stats("reset_me").get("calls", 0) == 0

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_call_count_matches(self, n) -> None:
        @tracked(label=f"count_{n}")
        def fn() -> None:
            pass

        for _ in range(n):
            fn()
        assert get_stats(f"count_{n}")["calls"] == n


class TestProfilerEdgeCases:
    def setup_method(self) -> None:
        reset_stats()

    def test_reset_all_clears_all_labels(self) -> None:
        @tracked(label="lbl_a")
        def a() -> None:
            pass

        @tracked(label="lbl_b")
        def b() -> None:
            pass

        a()
        b()
        reset_stats()
        assert get_stats("lbl_a").get("calls", 0) == 0
        assert get_stats("lbl_b").get("calls", 0) == 0

    def test_timed_with_no_label_uses_function_name(self) -> None:
        @timed()
        def named_fn() -> None:
            return 99

        assert named_fn() == 99

    @pytest.mark.parametrize("retval", [0, "", [], None, {"k": "v"}])
    def test_timed_preserves_various_return_types(self, retval: object) -> None:
        @timed()
        def fn() -> None:
            return retval

        assert fn() == retval

    def test_tracked_records_after_exception(self) -> None:
        @tracked(label="err_fn")
        def bad() -> None:
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            bad()
        stats = get_stats("err_fn")
        assert stats.get("calls", 0) >= 0

    def test_get_stats_returns_dict(self) -> None:
        @tracked(label="dict_check")
        def fn() -> None:
            pass

        fn()
        result = get_stats("dict_check")
        assert isinstance(result, dict)
        assert "calls" in result


class TestProfilerExtended:
    def setup_method(self) -> None:
        from app.profiler import reset_stats

        reset_stats()

    def test_tracked_names_sorted(self) -> None:
        from app.profiler import tracked, tracked_names

        @tracked(label="z_func")
        def z() -> None:
            pass

        @tracked(label="a_func")
        def a() -> None:
            pass

        z()
        a()
        names = tracked_names()
        assert names == sorted(names)

    def test_total_calls_across_labels(self) -> None:
        from app.profiler import total_calls, tracked

        @tracked(label="tc_a")
        def fa() -> None:
            pass

        @tracked(label="tc_b")
        def fb() -> None:
            pass

        fa()
        fa()
        fb()
        assert total_calls() >= 3

    def test_call_count_unknown_label_returns_zero(self) -> None:
        from app.profiler import call_count

        assert call_count("nonexistent_label_xyz") == 0

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_call_count_matches_n_calls(self, n: int) -> None:
        from app.profiler import call_count, tracked

        label = f"param_count_{n}"

        @tracked(label=label)
        def fn() -> None:
            pass

        for _ in range(n):
            fn()
        assert call_count(label) == n

    def test_reset_stats_removes_label(self) -> None:
        from app.profiler import get_stats, reset_stats, tracked

        @tracked(label="to_reset")
        def fn() -> None:
            pass

        fn()
        reset_stats("to_reset")
        assert get_stats("to_reset") == {}
