"""Asynchronous batch telemetry processing with chunked I/O."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


def chunk_readings(
    readings: list[dict[str, Any]],
    chunk_size: int = 25,
) -> Iterator[list[dict[str, Any]]]:
    """Yield successive chunks from a reading list.

    Args:
        readings: Full list of telemetry dicts.
        chunk_size: Maximum number of readings per chunk.

    Yields:
        Sublists of readings, each of length at most chunk_size.
    """
    for i in range(0, len(readings), chunk_size):
        yield readings[i : i + chunk_size]


def process_batch_with_chunking(
    readings: list[dict[str, Any]],
    process_fn: Any,
    chunk_size: int = 25,
) -> list[Any]:
    """Apply a processing function to readings in chunks.

    Useful when a single batch is too large to load into memory at once
    (e.g. large CSV uploads or queue flushes).

    Args:
        readings: Full list of telemetry dicts to process.
        process_fn: Callable that accepts a list of dicts and returns a list of results.
        chunk_size: Readings to include in each chunk.

    Returns:
        Flattened list of all results from all chunks.
    """
    all_results: list[Any] = []
    total_chunks = -(-len(readings) // chunk_size)  # ceiling division

    for chunk_index, chunk in enumerate(chunk_readings(readings, chunk_size), start=1):
        logger.debug("Processing chunk %d/%d (%d readings)", chunk_index, total_chunks, len(chunk))
        try:
            results = process_fn(chunk)
            all_results.extend(results)
        except Exception as exc:
            logger.error("Chunk %d failed: %s — skipping", chunk_index, exc)

    return all_results


def summarise_batch_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics over a list of prediction result dicts.

    Args:
        results: List of prediction dicts (each must have 'maintenance_needed'
                 and 'anomaly_score').

    Returns:
        Dict with 'total', 'maintenance_flagged', 'avg_anomaly_score',
        and 'anomaly_types_found'.
    """
    if not results:
        return {"total": 0, "maintenance_flagged": 0}

    maintenance_flagged = sum(1 for r in results if r.get("maintenance_needed"))
    anomaly_scores = [r["anomaly_score"] for r in results if "anomaly_score" in r]
    anomaly_types = list({r["anomaly_type"] for r in results if r.get("anomaly_type")})

    return {
        "total": len(results),
        "maintenance_flagged": maintenance_flagged,
        "avg_anomaly_score": round(sum(anomaly_scores) / len(anomaly_scores), 4)
        if anomaly_scores
        else 0.0,
        "anomaly_types_found": anomaly_types,
    }


def filter_batch(
    readings: list[dict[str, Any]],
    predicate: Any,
) -> list[dict[str, Any]]:
    """Return only readings that satisfy predicate(reading) == True.

    Args:
        readings: Input list of telemetry dicts.
        predicate: Callable accepting a dict and returning bool.

    Returns:
        Filtered list of readings.
    """
    return [r for r in readings if predicate(r)]


def partition_batch(
    readings: list[dict[str, Any]],
    predicate: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split readings into (matching, not-matching) based on predicate.

    Args:
        readings: Input list of telemetry dicts.
        predicate: Callable accepting a dict and returning bool.

    Returns:
        Tuple of (matching_list, non_matching_list).
    """
    matching, rest = [], []
    for r in readings:
        (matching if predicate(r) else rest).append(r)
    return matching, rest


def batch_error_rate(results: list[dict[str, Any]]) -> float:
    """Return fraction of results that contain an 'error' key.

    Args:
        results: List of result dicts from process_batch_with_chunking.

    Returns:
        Error fraction in [0, 1].
    """
    if not results:
        return 0.0
    return sum(1 for r in results if "error" in r) / len(results)
