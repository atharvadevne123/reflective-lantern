"""Extended Pydantic schema tests with parametrize and validation edge cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ClaimRequest, ClaimResponse, HealthResponse, MetricsResponse


def _valid_claim(**overrides) -> dict:
    base = {
        "claim_id": "CLM-001",
        "claim_age_days": 15.0,
        "policy_limit": 100_000.0,
        "claimed_amount": 20_000.0,
        "num_prior_claims": 1.0,
        "policy_tenure_years": 2.0,
        "geo_risk_tier": 2.5,
        "days_since_last_claim": 100.0,
        "policy_type": "auto",
    }
    base.update(overrides)
    return base


def test_claim_request_valid() -> None:
    req = ClaimRequest(**_valid_claim())
    assert req.claim_id == "CLM-001"


@pytest.mark.parametrize("policy_type", ["home", "auto", "life", "commercial"])
def test_claim_request_valid_policy_types(policy_type: str) -> None:
    req = ClaimRequest(**_valid_claim(policy_type=policy_type))
    assert req.policy_type == policy_type


@pytest.mark.parametrize("bad_type", ["boat", "flood", "pet", "HEALTH", ""])
def test_claim_request_invalid_policy_type(bad_type: str) -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(policy_type=bad_type))


def test_claim_request_policy_type_case_insensitive() -> None:
    req = ClaimRequest(**_valid_claim(policy_type="HOME"))
    assert req.policy_type == "home"


@pytest.mark.parametrize("bad_limit", [-1.0, 0.0])
def test_claim_request_non_positive_policy_limit(bad_limit: float) -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(policy_limit=bad_limit))


@pytest.mark.parametrize("bad_age", [-1.0, 36501.0])
def test_claim_request_bad_claim_age_days(bad_age: float) -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(claim_age_days=bad_age))


def test_claim_request_zero_age_valid() -> None:
    req = ClaimRequest(**_valid_claim(claim_age_days=0.0))
    assert req.claim_age_days == 0.0


@pytest.mark.parametrize("bad_geo", [0.9, 5.1, -1.0, 6.0])
def test_claim_request_bad_geo_risk_tier(bad_geo: float) -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(geo_risk_tier=bad_geo))


@pytest.mark.parametrize("valid_geo", [1.0, 2.5, 3.0, 5.0])
def test_claim_request_valid_geo_risk_tier(valid_geo: float) -> None:
    req = ClaimRequest(**_valid_claim(geo_risk_tier=valid_geo))
    assert req.geo_risk_tier == valid_geo


def test_claim_request_claimed_amount_exceeds_limit_times_1_5() -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(policy_limit=100_000.0, claimed_amount=160_000.0))


def test_claim_request_claimed_amount_at_limit_1_5x_valid() -> None:
    req = ClaimRequest(**_valid_claim(policy_limit=100_000.0, claimed_amount=150_000.0))
    assert req.claimed_amount == 150_000.0


def test_claim_request_num_prior_claims_negative_invalid() -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(num_prior_claims=-1.0))


def test_claim_request_num_prior_claims_above_100_invalid() -> None:
    with pytest.raises(ValidationError):
        ClaimRequest(**_valid_claim(num_prior_claims=101.0))


def test_claim_response_instantiation() -> None:
    resp = ClaimResponse(
        claim_id="CLM-001",
        severity_class=1,
        severity_label="medium",
        severity_score=0.65,
        probabilities={"low": 0.1, "medium": 0.65, "high": 0.2, "critical": 0.05},
        anomaly_flag=0,
        triage_priority=2,
    )
    assert resp.severity_label == "medium"
    assert resp.model_version == "1.0.0"


def test_health_response_defaults() -> None:
    h = HealthResponse(status="ok", model_loaded=True, db_connected=True)
    assert h.version == "1.0.0"


def test_metrics_response_nullable_fields() -> None:
    m = MetricsResponse(
        f1_macro_mean=None,
        f1_macro_std=None,
        n_features=None,
        n_samples=None,
        severity_classes=["low", "medium", "high", "critical"],
    )
    assert m.f1_macro_mean is None
    assert len(m.severity_classes) == 4
