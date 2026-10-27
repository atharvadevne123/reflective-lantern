"""Extended tests for ticket-oracle app/priority_rules.py."""

from __future__ import annotations

import pytest
from app.priority_rules import (
    P1_KEYWORDS,
    P1_TIERS,
    apply_priority_rules,
    estimate_resolution_hours,
    severity_label,
)


class TestConstants:
    def test_p1_keywords_is_frozenset(self) -> None:
        assert isinstance(P1_KEYWORDS, frozenset)

    def test_p1_keywords_non_empty(self) -> None:
        assert len(P1_KEYWORDS) > 0

    def test_p1_tiers_contains_gold(self) -> None:
        assert "Gold" in P1_TIERS


class TestApplyPriorityRules:
    def test_p1_keyword_overrides_to_p1(self) -> None:
        pred = {"priority": "P3", "sla_breach_risk": 0.1}
        payload = {"description": "Production down due to outage", "customer_tier": "Silver"}
        result = apply_priority_rules(pred, payload)
        assert result["priority"] == "P1"

    def test_rule_override_key_set(self) -> None:
        pred = {"priority": "P3", "sla_breach_risk": 0.1}
        payload = {"description": "outage affecting all users", "customer_tier": "Silver"}
        result = apply_priority_rules(pred, payload)
        assert result.get("rule_override") == "deterministic_p1_rule"

    def test_sla_risk_elevated_to_at_least_0_9(self) -> None:
        pred = {"priority": "P3", "sla_breach_risk": 0.1}
        payload = {"description": "database unreachable", "customer_tier": "Silver"}
        result = apply_priority_rules(pred, payload)
        assert result["sla_breach_risk"] >= 0.9

    def test_gold_outage_overrides_to_p1(self) -> None:
        pred = {"priority": "P2", "sla_breach_risk": 0.3}
        payload = {"description": "something minor", "customer_tier": "Gold", "incident_type": "outage"}
        result = apply_priority_rules(pred, payload)
        assert result["priority"] == "P1"

    def test_no_override_for_normal_ticket(self) -> None:
        pred = {"priority": "P3", "sla_breach_risk": 0.2}
        payload = {"description": "UI glitch on dashboard", "customer_tier": "Silver"}
        result = apply_priority_rules(pred, payload)
        assert result["priority"] == "P3"
        assert "rule_override" not in result

    def test_already_p1_stays_p1(self) -> None:
        pred = {"priority": "P1", "sla_breach_risk": 0.95}
        payload = {"description": "production down", "customer_tier": "Gold"}
        result = apply_priority_rules(pred, payload)
        assert result["priority"] == "P1"

    def test_returns_dict(self) -> None:
        pred = {"priority": "P2", "sla_breach_risk": 0.5}
        result = apply_priority_rules(pred, {"description": ""})
        assert isinstance(result, dict)

    def test_case_insensitive_keyword_match(self) -> None:
        pred = {"priority": "P4", "sla_breach_risk": 0.1}
        payload = {"description": "OUTAGE reported at data center"}
        result = apply_priority_rules(pred, payload)
        assert result["priority"] == "P1"


class TestSeverityLabel:
    def test_p1_is_critical(self) -> None:
        assert severity_label("P1") == "Critical"

    def test_p2_is_high(self) -> None:
        assert severity_label("P2") == "High"

    def test_p3_is_medium(self) -> None:
        assert severity_label("P3") == "Medium"

    def test_p4_is_low(self) -> None:
        assert severity_label("P4") == "Low"

    def test_unknown_priority(self) -> None:
        assert severity_label("P99") == "Unknown"


class TestEstimateResolutionHours:
    def test_p1_low_load(self) -> None:
        hours = estimate_resolution_hours("P1", 0.0)
        assert hours == pytest.approx(1.0)

    def test_p4_low_load(self) -> None:
        hours = estimate_resolution_hours("P4", 0.0)
        assert hours == pytest.approx(72.0)

    def test_high_load_increases_time(self) -> None:
        low_load = estimate_resolution_hours("P2", 0.0)
        high_load = estimate_resolution_hours("P2", 1.0)
        assert high_load > low_load

    def test_load_below_threshold_no_increase(self) -> None:
        base = estimate_resolution_hours("P3", 0.0)
        at_threshold = estimate_resolution_hours("P3", 0.7)
        assert at_threshold == pytest.approx(base)

    def test_unknown_priority_defaults_to_24h(self) -> None:
        assert estimate_resolution_hours("P99", 0.0) == pytest.approx(24.0)
