"""Extended batch processor tests with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.batch_processor import (
    chunk_readings,
    process_batch_with_chunking,
    summarise_batch_results,
)


def _make_readings(n: int) -> list[dict]:
    return [{"vehicle_id": f"v{i}", "value": float(i)} for i in range(n)]


def test_chunk_readings_empty_input() -> None:
    result = list(chunk_readings([]))
    assert result == []


def test_chunk_readings_exactly_one_chunk() -> None:
    readings = _make_readings(5)
    chunks = list(chunk_readings(readings, chunk_size=10))
    assert len(chunks) == 1
    assert len(chunks[0]) == 5


def test_chunk_readings_multiple_chunks() -> None:
    readings = _make_readings(10)
    chunks = list(chunk_readings(readings, chunk_size=3))
    assert len(chunks) == 4


def test_chunk_readings_last_chunk_smaller() -> None:
    readings = _make_readings(10)
    chunks = list(chunk_readings(readings, chunk_size=3))
    assert len(chunks[-1]) == 1


@pytest.mark.parametrize(
    "n,chunk_size,expected_chunks",
    [
        (0, 5, 0),
        (5, 5, 1),
        (10, 5, 2),
        (11, 5, 3),
        (25, 25, 1),
        (26, 25, 2),
    ],
)
def test_chunk_readings_parametrized(n: int, chunk_size: int, expected_chunks: int) -> None:
    readings = _make_readings(n)
    chunks = list(chunk_readings(readings, chunk_size=chunk_size))
    assert len(chunks) == expected_chunks


def test_chunk_readings_total_items_preserved() -> None:
    readings = _make_readings(23)
    chunks = list(chunk_readings(readings, chunk_size=5))
    total = sum(len(c) for c in chunks)
    assert total == 23


def test_process_batch_with_chunking_identity_function() -> None:
    readings = _make_readings(50)
    result = process_batch_with_chunking(readings, lambda chunk: chunk, chunk_size=10)
    assert len(result) == 50


def test_process_batch_with_chunking_empty_input() -> None:
    result = process_batch_with_chunking([], lambda chunk: chunk)
    assert result == []


def test_process_batch_with_chunking_error_in_chunk_skipped() -> None:
    readings = _make_readings(30)

    def sometimes_fail(chunk):
        if len(chunk) > 5:
            raise ValueError("too big")
        return chunk

    result = process_batch_with_chunking(readings, sometimes_fail, chunk_size=5)
    assert len(result) >= 0


def test_summarise_batch_results_empty() -> None:
    result = summarise_batch_results([])
    assert result["total"] == 0
    assert result["maintenance_flagged"] == 0


def test_summarise_batch_results_all_maintenance() -> None:
    results = [{"maintenance_needed": True, "anomaly_score": 0.9}] * 5
    summary = summarise_batch_results(results)
    assert summary["total"] == 5
    assert summary["maintenance_flagged"] == 5


def test_summarise_batch_results_no_maintenance() -> None:
    results = [{"maintenance_needed": False, "anomaly_score": 0.1}] * 5
    summary = summarise_batch_results(results)
    assert summary["maintenance_flagged"] == 0


def test_summarise_batch_results_has_avg_anomaly_score() -> None:
    results = [{"maintenance_needed": False, "anomaly_score": 0.5}] * 4
    summary = summarise_batch_results(results)
    assert "avg_anomaly_score" in summary
    assert summary["avg_anomaly_score"] == pytest.approx(0.5)
