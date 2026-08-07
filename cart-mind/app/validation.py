"""Input validation helpers beyond what Pydantic field constraints express.

Pydantic covers per-field ranges declaratively. What it does not cover is
cross-field coherence — a shopper who registered 30 days ago cannot have last
purchased 200 days ago — and those contradictions are exactly the ones that
produce confidently wrong predictions, because the model was never trained on
states that cannot physically occur.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationIssue(Exception):
    """Raised when a payload is internally inconsistent.

    Args:
        message: Human-readable description of the inconsistency.
        field: Name of the offending field, when a single field is at fault.
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


def check_temporal_coherence(payload: dict[str, Any]) -> list[str]:
    """Check that time-based fields describe a physically possible history.

    Args:
        payload: Raw request payload as a mapping.

    Returns:
        A list of human-readable warnings; empty when the payload is coherent.
    """
    warnings: list[str] = []
    since_reg = payload.get("days_since_registration")
    since_purchase = payload.get("days_since_last_purchase")

    if since_reg is not None and since_purchase is not None and since_purchase > since_reg:
        warnings.append(
            f"days_since_last_purchase ({since_purchase}) exceeds "
            f"days_since_registration ({since_reg}): a purchase cannot predate signup"
        )

    purchases = payload.get("purchase_count")
    if (
        purchases is not None
        and purchases > 0
        and since_purchase is not None
        and since_purchase >= 3650
    ):
        warnings.append("purchase_count is positive but the last purchase is over a decade old")
    return warnings


def check_engagement_coherence(payload: dict[str, Any]) -> list[str]:
    """Check that engagement counters are mutually consistent.

    A click implies a view, so a payload with clicks but no views describes a
    funnel that cannot have happened and signals upstream instrumentation drift.

    Args:
        payload: Raw request payload as a mapping.

    Returns:
        A list of human-readable warnings; empty when the payload is coherent.
    """
    warnings: list[str] = []
    views = payload.get("view_count")
    clicks = payload.get("click_count")

    if views is not None and clicks is not None and clicks > views:
        warnings.append(
            f"click_count ({clicks}) exceeds view_count ({views}): every click implies a view"
        )
    return warnings


def check_catalog_coherence(payload: dict[str, Any]) -> list[str]:
    """Check that item catalogue fields are mutually consistent.

    Args:
        payload: Raw request payload as a mapping.

    Returns:
        A list of human-readable warnings; empty when the payload is coherent.
    """
    warnings: list[str] = []
    rating = payload.get("item_avg_rating")
    reviews = payload.get("item_review_count")

    if rating is not None and reviews is not None and reviews == 0 and rating > 0:
        warnings.append("item_avg_rating is positive but item_review_count is zero")
    return warnings


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Run every coherence check and collect the warnings.

    Warnings are advisory: the request still gets scored, because rejecting a
    prediction outright over a soft inconsistency is worse for a serving path
    than scoring it and recording that the inputs looked odd.

    Args:
        payload: Raw request payload as a mapping.

    Returns:
        Combined list of warnings from all coherence checks.
    """
    warnings = (
        check_temporal_coherence(payload)
        + check_engagement_coherence(payload)
        + check_catalog_coherence(payload)
    )
    if warnings:
        logger.warning("Payload coherence warnings: %s", "; ".join(warnings))
    return warnings


def validate_batch(payloads: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Run coherence checks on a batch of payloads.

    Args:
        payloads: List of raw request payloads.

    Returns:
        Mapping from payload index (as string) to its warning list; only
        payloads with at least one warning are included.
    """
    results: dict[str, list[str]] = {}
    for i, payload in enumerate(payloads):
        warnings = validate_payload(payload)
        if warnings:
            results[str(i)] = warnings
    if results:
        logger.info("Batch validation: %d/%d payloads have warnings", len(results), len(payloads))
    return results
