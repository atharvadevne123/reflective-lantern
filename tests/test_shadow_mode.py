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
        for i in range(4):
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
