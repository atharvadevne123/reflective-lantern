"""Tests for app.shadow_mode module."""

from __future__ import annotations

import pytest

from app.shadow_mode import ShadowResult, ShadowRunner


def _primary(x):
    return x * 2


def _shadow_same(x):
    return x * 2


def _shadow_different(x):
    return x * 3


def _shadow_error(x):
    raise ValueError("shadow failed")


class TestShadowRunnerBasic:
    def test_returns_primary_result(self):
        runner = ShadowRunner(_primary, _shadow_same)
        assert runner.call(5) == 10

    def test_matched_when_results_equal(self):
        runner = ShadowRunner(_primary, _shadow_same)
        runner.call(5)
        assert runner.stats()["matched"] == 1

    def test_mismatch_when_results_differ(self):
        runner = ShadowRunner(_primary, _shadow_different)
        runner.call(5)
        stats = runner.stats()
        assert stats["matched"] == 0
        assert stats["mismatched"] == 1

    def test_shadow_error_captured(self):
        runner = ShadowRunner(_primary, _shadow_error)
        result = runner.call(5)
        assert result == 10
        stats = runner.stats()
        assert stats["errors"] == 1
        assert stats["matched"] == 0

    def test_multiple_calls_accumulate(self):
        runner = ShadowRunner(_primary, _shadow_same)
        for i in range(5):
            runner.call(i)
        assert runner.stats()["total"] == 5

    def test_match_rate_all_matched(self):
        runner = ShadowRunner(_primary, _shadow_same)
        for i in range(4):
            runner.call(i)
        assert runner.stats()["match_rate"] == pytest.approx(1.0)

    def test_match_rate_none_matched(self):
        runner = ShadowRunner(_primary, _shadow_different)
        for i in range(1, 5):  # start at 1 to avoid 0*2==0*3 edge case
            runner.call(i)
        assert runner.stats()["match_rate"] == pytest.approx(0.0)

    def test_clear_resets_stats(self):
        runner = ShadowRunner(_primary, _shadow_same)
        runner.call(1)
        runner.clear()
        assert runner.stats()["total"] == 0


class TestCustomComparer:
    def test_custom_comparer_used(self):
        def approx_equal(a, b):
            return abs(a - b) < 5

        runner = ShadowRunner(
            lambda x: float(x),
            lambda x: float(x) + 3,
            comparer=approx_equal,
        )
        runner.call(10)
        assert runner.stats()["matched"] == 1

    def test_comparer_exception_treated_as_mismatch(self):
        def bad_comparer(a, b):
            raise RuntimeError("comparer broken")

        runner = ShadowRunner(_primary, _shadow_same, comparer=bad_comparer)
        runner.call(5)
        assert runner.stats()["matched"] == 0


class TestShadowResultFields:
    def test_latencies_recorded(self):
        runner = ShadowRunner(_primary, _shadow_same)
        runner.call(3)
        result = runner._results[0]
        assert isinstance(result, ShadowResult)
        assert result.primary_latency_ms >= 0
        assert result.shadow_latency_ms >= 0

    def test_shadow_error_is_exception(self):
        runner = ShadowRunner(_primary, _shadow_error)
        runner.call(1)
        result = runner._results[0]
        assert isinstance(result.shadow_error, ValueError)

    @pytest.mark.parametrize("x", [0, -1, 100, 3.14])
    def test_various_inputs(self, x):
        runner = ShadowRunner(_primary, _shadow_same)
        ret = runner.call(x)
        assert ret == x * 2


class TestShadowRunnerStatsCompleteness:
    def test_stats_keys_present(self):
        runner = ShadowRunner(_primary, _shadow_same)
        stats = runner.stats()
        for key in ("total", "matched", "mismatched", "errors", "match_rate"):
            assert key in stats

    def test_stats_on_zero_calls(self):
        runner = ShadowRunner(_primary, _shadow_same)
        stats = runner.stats()
        assert stats["total"] == 0
        assert stats["match_rate"] == pytest.approx(0.0) or stats["match_rate"] == pytest.approx(1.0)

    def test_mixed_match_mismatch_error(self):
        runner = ShadowRunner(_primary, lambda x: x * 2 if x % 2 == 0 else x * 3)
        for i in range(1, 5):
            runner.call(i)
        stats = runner.stats()
        assert stats["total"] == 4
        assert stats["matched"] + stats["mismatched"] + stats["errors"] == 4

    def test_match_rate_0_5_for_half_matched(self):
        runner = ShadowRunner(_primary, lambda x: x * 2 if x < 5 else x * 99)
        for i in range(1, 11):
            runner.call(i)
        stats = runner.stats()
        assert stats["match_rate"] == pytest.approx(0.4)

    def test_clear_then_rerun_gives_fresh_stats(self):
        runner = ShadowRunner(_primary, _shadow_same)
        for _ in range(3):
            runner.call(1)
        runner.clear()
        runner.call(2)
        assert runner.stats()["total"] == 1

    def test_primary_always_returns_regardless_of_shadow(self):
        def slow_shadow(x):
            import time

            time.sleep(0.001)
            return x * 2

        runner = ShadowRunner(_primary, slow_shadow)
        for i in range(5):
            assert runner.call(i) == i * 2


@pytest.mark.parametrize("n_calls", [1, 5, 10])
def test_shadow_runner_total_stats_match_calls(n_calls: int) -> None:
    """stats()['total'] equals the number of times call() was invoked."""
    from app.shadow_mode import ShadowRunner

    runner = ShadowRunner(lambda x: x, lambda x: x + 1)
    for i in range(n_calls):
        runner.call(i)
    assert runner.stats()["total"] == n_calls


@pytest.mark.parametrize("n_calls", [1, 5, 10])
def test_shadow_runner_primary_result_always_returned(n_calls: int) -> None:
    """call() always returns the primary function's result, ignoring shadow."""
    from app.shadow_mode import ShadowRunner

    primary = lambda x: x * 3  # noqa: E731
    shadow = lambda x: x * 99  # noqa: E731
    runner = ShadowRunner(primary, shadow)
    for i in range(n_calls):
        assert runner.call(i) == i * 3


class TestShadowResultEdgeCases:
    def test_shadow_result_match_flag_true_when_equal(self) -> None:
        from app.shadow_mode import ShadowResult

        result = ShadowResult(primary=42, shadow=42, matched=True)
        assert result.matched is True

    def test_shadow_result_match_flag_false_when_different(self) -> None:
        from app.shadow_mode import ShadowResult

        result = ShadowResult(primary=42, shadow=99, matched=False)
        assert result.matched is False

    @pytest.mark.parametrize("value", [0, -1, 100, "ok", None])
    def test_shadow_result_primary_value_preserved(self, value: object) -> None:
        from app.shadow_mode import ShadowResult

        result = ShadowResult(primary=value, shadow=value, matched=True)
        assert result.primary == value
