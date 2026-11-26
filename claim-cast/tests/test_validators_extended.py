"""Extended validator tests with parametrize and edge cases."""

from __future__ import annotations

import pytest

from app.validators import is_high_risk_claim, validate_claim_data


def _valid_claim(**overrides) -> dict:
    base = {
        "claimed_amount": 20_000.0,
        "policy_limit": 100_000.0,
        "claim_age_days": 10.0,
        "policy_tenure_years": 2.0,
        "policy_type": "auto",
        "geo_risk_tier": 3.0,
        "num_prior_claims": 1.0,
    }
    base.update(overrides)
    return base


def test_valid_claim_no_errors() -> None:
    errors = validate_claim_data(_valid_claim())
    assert errors == []


@pytest.mark.parametrize("bad_amount", [0.0, -1.0, -100_000.0])
def test_zero_or_negative_claimed_amount(bad_amount: float) -> None:
    errors = validate_claim_data(_valid_claim(claimed_amount=bad_amount))
    assert any("claimed_amount" in e for e in errors)


@pytest.mark.parametrize("bad_limit", [0.0, -50_000.0])
def test_zero_or_negative_policy_limit(bad_limit: float) -> None:
    errors = validate_claim_data(_valid_claim(policy_limit=bad_limit))
    assert any("policy_limit" in e for e in errors)


@pytest.mark.parametrize("bad_age", [-1.0, -365.0])
def test_negative_claim_age_days(bad_age: float) -> None:
    errors = validate_claim_data(_valid_claim(claim_age_days=bad_age))
    assert any("claim_age_days" in e for e in errors)


def test_zero_claim_age_is_valid() -> None:
    errors = validate_claim_data(_valid_claim(claim_age_days=0.0))
    assert not any("claim_age_days" in e for e in errors)


@pytest.mark.parametrize("bad_tenure", [0.0, -1.0])
def test_non_positive_policy_tenure(bad_tenure: float) -> None:
    errors = validate_claim_data(_valid_claim(policy_tenure_years=bad_tenure))
    assert any("policy_tenure_years" in e for e in errors)


@pytest.mark.parametrize("valid_type", ["home", "auto", "life", "commercial"])
def test_valid_policy_types(valid_type: str) -> None:
    errors = validate_claim_data(_valid_claim(policy_type=valid_type))
    assert not any("policy_type" in e for e in errors)


@pytest.mark.parametrize("invalid_type", ["boat", "pet", "flood", "", "HOME", "AUTO"])
def test_invalid_policy_types(invalid_type: str) -> None:
    errors = validate_claim_data(_valid_claim(policy_type=invalid_type))
    assert any("policy_type" in e for e in errors)


@pytest.mark.parametrize("valid_geo", [1.0, 2.5, 3.0, 5.0])
def test_valid_geo_risk_tier(valid_geo: float) -> None:
    errors = validate_claim_data(_valid_claim(geo_risk_tier=valid_geo))
    assert not any("geo_risk_tier" in e for e in errors)


@pytest.mark.parametrize("invalid_geo", [0.0, 0.9, 5.1, 6.0, -1.0])
def test_invalid_geo_risk_tier(invalid_geo: float) -> None:
    errors = validate_claim_data(_valid_claim(geo_risk_tier=invalid_geo))
    assert any("geo_risk_tier" in e for e in errors)


def test_multiple_errors_returned() -> None:
    bad = _valid_claim(claimed_amount=-1.0, policy_limit=-1.0, policy_type="unknown")
    errors = validate_claim_data(bad)
    assert len(errors) >= 3


def test_validate_returns_list_type() -> None:
    result = validate_claim_data(_valid_claim())
    assert isinstance(result, list)


@pytest.mark.parametrize(
    "claimed,limit,prior,expected",
    [
        (70_000.0, 100_000.0, 0.0, True),
        (50_000.0, 100_000.0, 0.0, False),
        (10_000.0, 100_000.0, 4.0, True),
        (10_000.0, 100_000.0, 3.0, False),
        (60_000.0, 100_000.0, 1.0, False),
        (61_000.0, 100_000.0, 0.0, True),
    ],
)
def test_is_high_risk_claim(claimed: float, limit: float, prior: float, expected: bool) -> None:
    assert is_high_risk_claim(claimed, limit, prior) is expected


def test_is_high_risk_zero_policy_limit() -> None:
    result = is_high_risk_claim(10_000.0, 0.0, 0.0)
    assert isinstance(result, bool)


def test_is_high_risk_very_large_claim() -> None:
    assert is_high_risk_claim(1_000_000.0, 100_000.0, 0.0) is True


def test_is_high_risk_zero_claim() -> None:
    assert is_high_risk_claim(0.0, 100_000.0, 0.0) is False
