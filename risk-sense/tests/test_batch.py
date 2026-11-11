"""Tests for batch scoring utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.batch import chunk_list, summarise_batch_results


class TestChunkList:
    def test_even_chunks(self) -> None:
        items = list(range(10))
        chunks = chunk_list(items, 5)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3, 4]

    def test_uneven_chunks(self) -> None:
        items = list(range(7))
        chunks = chunk_list(items, 3)
        assert len(chunks) == 3
        assert chunks[-1] == [6]

    def test_empty_list(self) -> None:
        assert chunk_list([], 5) == []

    def test_chunk_size_larger_than_list(self) -> None:
        items = [1, 2, 3]
        chunks = chunk_list(items, 10)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_list([1, 2, 3], 0)

    @pytest.mark.parametrize("size,chunk", [(100, 10), (50, 7), (1, 1)])
    def test_total_items_preserved(self, size: int, chunk: int) -> None:
        items = list(range(size))
        chunks = chunk_list(items, chunk)
        total = sum(len(c) for c in chunks)
        assert total == size


class TestSummariseBatchResults:
    def test_empty_scores(self) -> None:
        result = summarise_batch_results([])
        assert result["total"] == 0
        assert result["scored"] == 0

    def test_all_valid_scores(self) -> None:
        scores = [0.1, 0.3, 0.6, 0.8, 0.2]
        result = summarise_batch_results(scores)
        assert result["total"] == 5
        assert result["scored"] == 5
        assert result["failed"] == 0
        assert result["high_risk_count"] == 2

    def test_nan_scores_counted_as_failed(self) -> None:
        scores = [0.3, float("nan"), 0.7, float("nan")]
        result = summarise_batch_results(scores)
        assert result["failed"] == 2
        assert result["scored"] == 2

    def test_all_nan_scores(self) -> None:
        scores = [float("nan")] * 3
        result = summarise_batch_results(scores)
        assert result["scored"] == 0
        assert result["failed"] == 3

    def test_high_risk_rate_calculation(self) -> None:
        scores = [0.6, 0.7, 0.3, 0.1]  # 2/4 = 0.5
        result = summarise_batch_results(scores)
        assert result["high_risk_rate"] == pytest.approx(0.5)
