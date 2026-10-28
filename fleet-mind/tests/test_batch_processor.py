"""Tests for chunked batch processing utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.batch_processor import (
    chunk_readings,
    process_batch_with_chunking,
    summarise_batch_results,
)


@pytest.fixture
def sample_results():
    return [
        {"maintenance_needed": True, "anomaly_score": 0.8, "anomaly_type": "engine_overheat"},
        {"maintenance_needed": False, "anomaly_score": 0.1, "anomaly_type": None},
        {"maintenance_needed": True, "anomaly_score": 0.7, "anomaly_type": "brake_worn"},
    ]


class TestChunkReadings:
    def test_single_chunk_when_small(self):
        readings = [{"a": i} for i in range(5)]
        chunks = list(chunk_readings(readings, chunk_size=10))
        assert len(chunks) == 1

    @pytest.mark.parametrize("n,chunk_size,expected_chunks", [
        (10, 3, 4),
        (9, 3, 3),
        (0, 5, 0),
    ])
    def test_chunk_count(self, n, chunk_size, expected_chunks):
        readings = [{"v": i} for i in range(n)]
        chunks = list(chunk_readings(readings, chunk_size=chunk_size))
        assert len(chunks) == expected_chunks

    def test_all_items_preserved(self):
        readings = [{"i": i} for i in range(25)]
        all_items = [item for chunk in chunk_readings(readings, 7) for item in chunk]
        assert all_items == readings


class TestProcessBatchWithChunking:
    def test_processes_all_readings(self):
        readings = [{"v": i} for i in range(10)]
        results = process_batch_with_chunking(readings, lambda c: [{"r": x["v"]} for x in c], chunk_size=3)
        assert len(results) == 10

    def test_handles_empty_input(self):
        results = process_batch_with_chunking([], lambda c: c)
        assert results == []


class TestSummariseBatchResults:
    def test_totals(self, sample_results):
        summary = summarise_batch_results(sample_results)
        assert summary["total"] == 3

    def test_maintenance_flagged(self, sample_results):
        summary = summarise_batch_results(sample_results)
        assert summary["maintenance_flagged"] == 2

    def test_avg_anomaly_score_range(self, sample_results):
        summary = summarise_batch_results(sample_results)
        assert 0.0 <= summary["avg_anomaly_score"] <= 1.0

    def test_empty_results(self):
        summary = summarise_batch_results([])
        assert summary["total"] == 0
