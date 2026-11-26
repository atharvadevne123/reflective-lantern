"""Extended priority rules tests with parametrize and edge cases."""

from __future__ import annotations

import pytest

from app.priority_rules import P1_KEYWORDS, apply_priority_rules


def _base_prediction(**overrides) -> dict:
    pred = {"priority": "P3", "sla_breach_risk": 0.1}
    pred.update(overrides)
    return pred


def _base_payload(**overrides) -> dict:
    payload = {
        "description": "Normal ticket",
        "customer_tier": "Standard",
        "incident_type": "general",
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("keyword", list(P1_KEYWORDS)[:5])
def test_p1_keyword_forces_p1_priority(keyword: str) -> None:
    pred = _base_prediction()
    result = apply_priority_rules(pred, _base_payload(description=f"We have a {keyword} situation"))
    assert result["priority"] == "P1"


@pytest.mark.parametrize("keyword", list(P1_KEYWORDS))
def test_all_p1_keywords_trigger_override(keyword: str) -> None:
    pred = _base_prediction(priority="P4")
    result = apply_priority_rules(pred, _base_payload(description=keyword))
    assert result["priority"] == "P1"


def test_p1_keyword_case_insensitive() -> None:
    pred = _base_prediction()
    result = apply_priority_rules(pred, _base_payload(description="PRODUCTION DOWN now"))
    assert result["priority"] == "P1"


def test_no_p1_keyword_does_not_override() -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(pred, _base_payload(description="Login is slow"))
    assert result["priority"] == "P3"


def test_gold_tier_outage_forces_p1() -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(pred, _base_payload(customer_tier="Gold", incident_type="outage"))
    assert result["priority"] == "P1"


def test_gold_tier_security_incident_forces_p1() -> None:
    pred = _base_prediction(priority="P2")
    result = apply_priority_rules(
        pred, _base_payload(customer_tier="Gold", incident_type="security_incident")
    )
    assert result["priority"] == "P1"


def test_gold_tier_data_loss_forces_p1() -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(
        pred, _base_payload(customer_tier="Gold", incident_type="data_loss")
    )
    assert result["priority"] == "P1"


def test_silver_tier_outage_does_not_force_p1() -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(
        pred, _base_payload(customer_tier="Silver", incident_type="outage")
    )
    assert result["priority"] == "P3"


def test_gold_tier_general_incident_does_not_force_p1() -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(
        pred, _base_payload(customer_tier="Gold", incident_type="general")
    )
    assert result["priority"] == "P3"


def test_rule_override_flag_set_on_p1_trigger() -> None:
    pred = _base_prediction()
    result = apply_priority_rules(pred, _base_payload(description="production down"))
    assert result.get("rule_override") == "deterministic_p1_rule"


def test_sla_breach_risk_elevated_on_p1_trigger() -> None:
    pred = _base_prediction(sla_breach_risk=0.1)
    result = apply_priority_rules(pred, _base_payload(description="outage detected"))
    assert result["sla_breach_risk"] >= 0.9


def test_already_p1_no_rule_override_added() -> None:
    pred = _base_prediction(priority="P1")
    result = apply_priority_rules(pred, _base_payload(description="outage detected"))
    assert result.get("rule_override") is None


def test_apply_priority_rules_returns_dict() -> None:
    result = apply_priority_rules(_base_prediction(), _base_payload())
    assert isinstance(result, dict)


def test_apply_priority_rules_mutates_in_place() -> None:
    pred = _base_prediction()
    result = apply_priority_rules(pred, _base_payload(description="outage"))
    assert result is pred


@pytest.mark.parametrize("empty_desc", ["", "   ", "Normal request", "Minor issue"])
def test_non_p1_descriptions_no_override(empty_desc: str) -> None:
    pred = _base_prediction(priority="P3")
    result = apply_priority_rules(pred, _base_payload(description=empty_desc))
    assert result["priority"] == "P3"
