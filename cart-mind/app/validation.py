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
        """Initialise the exception with a human-readable *message* and optional *field* name."""
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


def check_price_coherence(payload: dict[str, Any]) -> list[str]:
    """Check that pricing fields are internally consistent.

    Args:
        payload: Raw request payload as a mapping.

    Returns:
        List of human-readable warnings; empty when coherent.
    """
    warnings: list[str] = []
    discount_pct = payload.get("item_discount_pct")
    if discount_pct is not None:
        if discount_pct < 0:
            warnings.append(f"item_discount_pct is negative ({discount_pct})")
        if discount_pct > 100:
            warnings.append(f"item_discount_pct exceeds 100 ({discount_pct})")
    price = payload.get("item_price")
    avg_order = payload.get("avg_order_value")
    if price is not None and price < 0:
        warnings.append(f"item_price is negative ({price})")
    if avg_order is not None and avg_order < 0:
        warnings.append(f"avg_order_value is negative ({avg_order})")
    return warnings


def sanitise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with invalid numeric values clamped to valid ranges.

    Args:
        payload: Raw request payload.

    Returns:
        New dict with out-of-range values replaced by their boundary values.
    """
    out = dict(payload)
    if "item_discount_pct" in out and out["item_discount_pct"] is not None:
        out["item_discount_pct"] = max(0.0, min(100.0, float(out["item_discount_pct"])))
    if "item_price" in out and out["item_price"] is not None:
        out["item_price"] = max(0.0, float(out["item_price"]))
    if "avg_order_value" in out and out["avg_order_value"] is not None:
        out["avg_order_value"] = max(0.0, float(out["avg_order_value"]))
    if "cart_abandon_rate" in out and out["cart_abandon_rate"] is not None:
        out["cart_abandon_rate"] = max(0.0, min(1.0, float(out["cart_abandon_rate"])))
    return out


def validation_summary(results: dict[str, list[str]]) -> dict[str, Any]:
    """Summarise validate_batch output into aggregate statistics.

    Args:
        results: Mapping of index → warnings, as returned by ``validate_batch``.

    Returns:
        Dict with ``total_warnings``, ``affected_payloads``, and
        ``most_common_warning`` (or ``None`` if no warnings).
    """
    all_warnings = [w for warnings in results.values() for w in warnings]
    if not all_warnings:
        return {"total_warnings": 0, "affected_payloads": 0, "most_common_warning": None}
    freq: dict[str, int] = {}
    for w in all_warnings:
        freq[w] = freq.get(w, 0) + 1
    most_common = max(freq, key=lambda k: freq[k])
    return {
        "total_warnings": len(all_warnings),
        "affected_payloads": len(results),
        "most_common_warning": most_common,
    }


def price_discount_pct(original_price: float, discounted_price: float) -> float:
    """Return percentage discount from original to discounted price.

    Args:
        original_price: Original item price.
        discounted_price: Discounted price.

    Returns:
        Discount percentage in [0, 100]. Returns 0.0 if original_price is zero.
    """
    if original_price <= 0.0:
        return 0.0
    discount = (original_price - discounted_price) / original_price * 100.0
    return round(max(0.0, min(100.0, discount)), 4)


def cart_value_tier(cart_value: float) -> str:
    """Categorise cart value into a spend tier.

    Args:
        cart_value: Total cart value.

    Returns:
        'low' (< 25), 'medium' (25-99), 'high' (100-499), or 'premium' (500+).
    """
    if cart_value < 25.0:
        return "low"
    if cart_value < 100.0:
        return "medium"
    if cart_value < 500.0:
        return "high"
    return "premium"


def item_count_flag(item_count: int, bulk_threshold: int = 10) -> bool:
    """Return True if the item count meets or exceeds the bulk threshold.

    Args:
        item_count: Number of items in the cart.
        bulk_threshold: Minimum count to be considered bulk.

    Returns:
        True if item_count >= bulk_threshold.
    """
    return item_count >= bulk_threshold


def cross_field_penalty_score(warnings: list[str]) -> float:
    """Convert a warning list to a numeric penalty for downstream scoring.

    Each coherence warning reduces the score by 0.1, floored at 0.0.

    Args:
        warnings: List of warning strings from any coherence check.

    Returns:
        Penalty score in [0.0, 1.0]; 0.0 means no penalty (no warnings).
    """
    return min(1.0, round(len(warnings) * 0.1, 4))


def normalise_payload(
    payload: dict[str, Any], defaults: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fill missing keys from *defaults* and return a complete payload copy.

    Args:
        payload: Raw incoming payload, possibly with missing optional fields.
        defaults: Fallback values for any missing keys. If ``None``, no filling
            is performed.

    Returns:
        New dict with defaults merged in where the original had no value.
    """
    if defaults is None:
        return dict(payload)
    result = dict(defaults)
    result.update(payload)
    return result
