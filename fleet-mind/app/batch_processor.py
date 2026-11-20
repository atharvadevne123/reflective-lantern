"""Asynchronous batch telemetry processing with chunked I/O."""

from __future__ import annotations

import logging
from typing import Any, Iterator

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
        "avg_anomaly_score": round(sum(anomaly_scores) / len(anomaly_scores), 4) if anomaly_scores else 0.0,
        "anomaly_types_found": anomaly_types,
    }
