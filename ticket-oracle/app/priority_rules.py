"""Deterministic override rules for ticket priority classification.

Business rules that must always fire regardless of model output —
e.g. any P1 keyword in description or Gold tier + outage always P1.
Applied after model inference so they override but never hide the
model's probability scores.
"""

from __future__ import annotations

P1_KEYWORDS = frozenset([
    "production down", "outage", "data loss", "security breach",
    "ransomware", "all users affected", "complete failure",
    "database unreachable", "cannot login", "breach",
])

P1_TIERS = frozenset(["Gold"])
P1_INCIDENT_TYPES = frozenset(["outage", "security_incident", "data_loss"])


def _contains_p1_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in P1_KEYWORDS)


def apply_priority_rules(
    prediction: dict,
    payload: dict,
) -> dict:
    """Apply deterministic overrides to a model prediction.

    Args:
        prediction: Model output dict with at least 'priority' key.
        payload: Raw ticket payload dict with description, customer_tier, etc.

    Returns:
        Updated prediction dict (mutated in-place and returned).
    """
    description = payload.get("description", "")
    tier = payload.get("customer_tier", "")
    incident_type = payload.get("incident_type", "")

    forced_p1 = (
        _contains_p1_keyword(description)
        or (tier in P1_TIERS and incident_type in P1_INCIDENT_TYPES)
    )

    if forced_p1 and prediction.get("priority") != "P1":
        prediction["priority"] = "P1"
        prediction["rule_override"] = "deterministic_p1_rule"
        prediction["sla_breach_risk"] = max(prediction.get("sla_breach_risk", 0.0), 0.9)

    return prediction


def severity_label(priority: str) -> str:
    """Map priority code to human-readable severity string."""
    return {
        "P1": "Critical",
        "P2": "High",
        "P3": "Medium",
        "P4": "Low",
    }.get(priority, "Unknown")


def estimate_resolution_hours(priority: str, agent_load: float) -> float:
    """Estimate resolution time in hours given priority and current agent load.

    Args:
        priority: One of P1-P4.
        agent_load: Fraction 0.0-1.0 of agent capacity utilised.

    Returns:
        Estimated resolution hours (float).
    """
    base = {"P1": 1.0, "P2": 4.0, "P3": 24.0, "P4": 72.0}.get(priority, 24.0)
    load_multiplier = 1.0 + max(0.0, agent_load - 0.7) * 2.0
    return round(base * load_multiplier, 2)
