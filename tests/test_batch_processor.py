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


class TestBatchProcessorStream:
    def test_run_stream_yields_all_batches(self):
        bp = BatchProcessor(_identity, batch_size=3)
        batches = list(bp.run_stream(list(range(7))))
        assert len(batches) == 3
        assert batches[0].batch_index == 0
        assert batches[2].batch_index == 2

    def test_run_stream_results_correct(self):
        bp = BatchProcessor(_double, batch_size=2)
        results = [r for br in bp.run_stream([1, 2, 3, 4]) for r in br.results]
        assert results == [2, 4, 6, 8]

    def test_run_stream_empty(self):
        bp = BatchProcessor(_identity, batch_size=5)
        assert list(bp.run_stream([])) == []

    def test_batch_count_helper(self):
        bp = BatchProcessor(_identity, batch_size=10)
        assert bp.batch_count(0) == 0
        assert bp.batch_count(10) == 1
        assert bp.batch_count(11) == 2
        assert bp.batch_count(25) == 3

    def test_run_stream_collect_errors(self):
        bp = BatchProcessor(_failing, batch_size=5, error_handling="collect")
        batches = list(bp.run_stream(list(range(10))))
        assert all(len(br.errors) == 1 for br in batches)
