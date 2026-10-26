"""Async-friendly batch prediction processor for Ticket-Oracle.

Handles chunked inference over a list of TicketPayload objects,
collecting results and per-ticket errors without aborting the whole batch.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def process_batch(
    tickets: list[dict[str, Any]],
    predict_fn,
    chunk_size: int = 25,
) -> dict[str, Any]:
    """Run predict_fn over tickets in chunks, collecting results and errors.

    Args:
        tickets: List of raw ticket dicts (already validated by the caller).
        predict_fn: Callable accepting a single dict and returning a prediction dict.
        chunk_size: Max items per chunk; limits peak memory usage.

    Returns:
        Dict with 'results' (list of prediction dicts) and 'errors' (list of
        {ticket_id, error} for any failed items).
    """
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for start in range(0, len(tickets), chunk_size):
        chunk = tickets[start : start + chunk_size]
        for ticket in chunk:
            ticket_id = ticket.get("ticket_id", "unknown")
            try:
                result = predict_fn(ticket)
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "batch_item_failed",
                    extra={"ticket_id": ticket_id, "error": str(exc)},
                )
                errors.append({"ticket_id": ticket_id, "error": str(exc)})

    logger.info(
        "batch_processed",
        extra={"total": len(tickets), "success": len(results), "errors": len(errors)},
    )
    return {"results": results, "errors": errors}


def aggregate_batch_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics over a batch of prediction results.

    Args:
        results: List of prediction dicts from process_batch.

    Returns:
        Dict with priority distribution and average SLA breach risk.
    """
    if not results:
        return {"priority_counts": {}, "avg_sla_breach_risk": 0.0, "total": 0}

    priority_counts: dict[str, int] = {}
    total_sla_risk = 0.0

    for r in results:
        p = r.get("priority", "unknown")
        priority_counts[p] = priority_counts.get(p, 0) + 1
        total_sla_risk += float(r.get("sla_breach_risk", 0.0))

    return {
        "priority_counts": priority_counts,
        "avg_sla_breach_risk": round(total_sla_risk / len(results), 4),
        "total": len(results),
    }
