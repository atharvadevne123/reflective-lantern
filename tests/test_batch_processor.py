"""Tests for app.batch_processor module."""

from __future__ import annotations

import pytest

from app.batch_processor import BatchProcessor


def _identity(items):
    return items


def _double(items):
    return [x * 2 for x in items]


def _failing(items):
    raise RuntimeError("intentional failure")


class TestBatchProcessorBasic:
    def test_processes_all_items(self):
        bp = BatchProcessor(_identity, batch_size=10)
        summary = bp.run(list(range(25)))
        assert summary.total_items == 25
        assert summary.total_results == 25

    def test_batch_count_correct(self):
        bp = BatchProcessor(_identity, batch_size=10)
        summary = bp.run(list(range(25)))
        assert summary.total_batches == 3

    def test_empty_input(self):
        bp = BatchProcessor(_identity, batch_size=10)
        summary = bp.run([])
        assert summary.total_items == 0
        assert summary.total_batches == 0
        assert summary.total_results == 0

    def test_single_batch(self):
        bp = BatchProcessor(_double, batch_size=100)
        summary = bp.run([1, 2, 3])
        assert summary.total_results == 3
        assert summary.total_errors == 0

    @pytest.mark.parametrize("size", [1, 5, 100])
    def test_various_batch_sizes(self, size):
        bp = BatchProcessor(_identity, batch_size=size)
        summary = bp.run(list(range(50)))
        assert summary.total_results == 50


class TestBatchProcessorErrors:
    def test_error_handling_raise_propagates(self):
        bp = BatchProcessor(_failing, batch_size=5, error_handling="raise")
        with pytest.raises(RuntimeError, match="intentional"):
            bp.run(list(range(10)))

    def test_error_handling_collect_continues(self):
        bp = BatchProcessor(_failing, batch_size=5, error_handling="collect")
        summary = bp.run(list(range(10)))
        assert summary.total_errors == 2
        assert summary.total_results == 0

    def test_invalid_error_handling_raises(self):
        with pytest.raises(ValueError, match="error_handling"):
            BatchProcessor(_identity, error_handling="ignore")

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            BatchProcessor(_identity, batch_size=0)


class TestBatchProcessorCallbacks:
    def test_on_batch_done_called_per_batch(self):
        called = []
        bp = BatchProcessor(_identity, batch_size=5, on_batch_done=lambda br: called.append(br.batch_index))
        bp.run(list(range(15)))
        assert called == [0, 1, 2]

    def test_callback_receives_batch_result(self):
        results_seen = []
        bp = BatchProcessor(_double, batch_size=3, on_batch_done=lambda br: results_seen.extend(br.results))
        bp.run([1, 2, 3])
        assert results_seen == [2, 4, 6]

    def test_callback_exception_does_not_abort(self):
        def bad_cb(br):
            raise RuntimeError("cb fail")

        bp = BatchProcessor(_identity, batch_size=5, on_batch_done=bad_cb)
        summary = bp.run(list(range(10)))
        assert summary.total_results == 10


class TestBatchProcessorSummary:
    def test_summary_total_items_matches_input(self):
        bp = BatchProcessor(_identity, batch_size=7)
        summary = bp.run(list(range(21)))
        assert summary.total_items == 21

    def test_summary_no_errors_on_clean_run(self):
        bp = BatchProcessor(_identity, batch_size=5)
        summary = bp.run(list(range(20)))
        assert summary.total_errors == 0

    def test_batch_size_one_creates_n_batches(self):
        n = 10
        bp = BatchProcessor(_identity, batch_size=1)
        summary = bp.run(list(range(n)))
        assert summary.total_batches == n

    def test_results_count_equals_input_on_identity(self):
        bp = BatchProcessor(_identity, batch_size=4)
        items = list(range(13))
        summary = bp.run(items)
        assert summary.total_results == len(items)

    @pytest.mark.parametrize("n,size,expected_batches", [
        (10, 5, 2),
        (11, 5, 3),
        (100, 10, 10),
        (1, 100, 1),
    ])
    def test_batch_count_formula(self, n: int, size: int, expected_batches: int) -> None:
        bp = BatchProcessor(_identity, batch_size=size)
        summary = bp.run(list(range(n)))
        assert summary.total_batches == expected_batches
