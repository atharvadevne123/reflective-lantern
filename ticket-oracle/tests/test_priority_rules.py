"""Priority rules tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.priority_rules import apply_priority_rules, estimate_resolution_hours, severity_label

BASE_PAYLOAD = {
    "description": "Printer is not working on the second floor",
    "customer_tier": "Bronze",
    "incident_type": "hardware",
}


def test_no_override_for_low_priority():
    pred = {"priority": "P4", "sla_breach_risk": 0.1}
    result = apply_priority_rules(pred, BASE_PAYLOAD)
    assert result["priority"] == "P4"
    assert "rule_override" not in result


def test_p1_keyword_forces_override():
    payload = {**BASE_PAYLOAD, "description": "Production down, all users affected"}
    pred = {"priority": "P3", "sla_breach_risk": 0.2}
    result = apply_priority_rules(pred, payload)
    assert result["priority"] == "P1"
    assert result["rule_override"] == "deterministic_p1_rule"
    assert result["sla_breach_risk"] >= 0.9


def test_gold_tier_outage_forces_p1():
    payload = {**BASE_PAYLOAD, "customer_tier": "Gold", "incident_type": "outage"}
    pred = {"priority": "P2", "sla_breach_risk": 0.5}
    result = apply_priority_rules(pred, payload)
    assert result["priority"] == "P1"


def test_gold_tier_non_outage_no_override():
    payload = {**BASE_PAYLOAD, "customer_tier": "Gold", "incident_type": "request"}
    pred = {"priority": "P3", "sla_breach_risk": 0.1}
    result = apply_priority_rules(pred, payload)
    assert result["priority"] == "P3"


def test_already_p1_no_change():
    payload = {**BASE_PAYLOAD, "description": "outage"}
    pred = {"priority": "P1", "sla_breach_risk": 0.95}
    result = apply_priority_rules(pred, payload)
    assert result["priority"] == "P1"


def test_severity_labels():
    assert severity_label("P1") == "Critical"
    assert severity_label("P2") == "High"
    assert severity_label("P3") == "Medium"
    assert severity_label("P4") == "Low"
    assert severity_label("P9") == "Unknown"


def test_resolution_hours_p1_low_load():
    hours = estimate_resolution_hours("P1", 0.3)
    assert hours == pytest.approx(1.0)


def test_resolution_hours_p4_high_load():
    hours = estimate_resolution_hours("P4", 0.9)
    assert hours > 72.0


def test_resolution_hours_increases_with_load():
    low = estimate_resolution_hours("P2", 0.2)
    high = estimate_resolution_hours("P2", 0.95)
    assert high > low
