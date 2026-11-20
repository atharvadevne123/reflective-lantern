"""Input validation utilities for claim data quality checks."""
from __future__ import annotations

from typing import Any


def validate_claim_data(data: dict[str, Any]) -> list[str]:
    """Validate raw claim data and return list of validation errors.

    Args:
        data: Raw claim dictionary with claim fields.

    Returns:
        List of error strings. Empty list means data is valid.
    """
    errors: list[str] = []

    if data.get("claimed_amount", 0) <= 0:
        errors.append("claimed_amount must be positive")

    if data.get("policy_limit", 0) <= 0:
        errors.append("policy_limit must be positive")

    if data.get("claim_age_days", 0) < 0:
        errors.append("claim_age_days cannot be negative")

    if data.get("policy_tenure_years", 0) <= 0:
        errors.append("policy_tenure_years must be positive")

    policy_type = data.get("policy_type", "").lower()
    if policy_type not in {"home", "auto", "life", "commercial"}:
        errors.append(f"policy_type '{policy_type}' is invalid")

    geo_risk = data.get("geo_risk_tier", 0)
    if not (1.0 <= geo_risk <= 5.0):
        errors.append("geo_risk_tier must be between 1 and 5")

    return errors


def is_high_risk_claim(claimed_amount: float, policy_limit: float, num_prior_claims: float) -> bool:
    """Quick heuristic check for obviously high-risk claims.

    Args:
        claimed_amount: Amount claimed in USD.
        policy_limit: Policy coverage limit in USD.
        num_prior_claims: Number of prior claims on the policy.

    Returns:
        True if the claim matches high-risk criteria.
    """
    severity_ratio = claimed_amount / max(policy_limit, 1)
    return severity_ratio > 0.6 or num_prior_claims >= 4
