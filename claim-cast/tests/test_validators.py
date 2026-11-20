"""Tests for claim data validation utilities."""
from __future__ import annotations

import pytest

from app.validators import is_high_risk_claim, validate_claim_data


def _valid_data() -> dict:
    return {
        "claimed_amount": 25000.0,
        "policy_limit": 100000.0,
        "claim_age_days": 15.0,
        "policy_tenure_years": 3.0,
        "policy_type": "auto",
        "geo_risk_tier": 2.5,
    }


def test_validate_claim_data_valid():
    errors = validate_claim_data(_valid_data())
    assert errors == []


def test_validate_claim_data_negative_amount():
    data = {**_valid_data(), "claimed_amount": -100}
    errors = validate_claim_data(data)
    assert any("claimed_amount" in e for e in errors)


def test_validate_claim_data_invalid_policy_type():
    data = {**_valid_data(), "policy_type": "spaceship"}
    errors = validate_claim_data(data)
    assert any("policy_type" in e for e in errors)


def test_validate_claim_data_invalid_geo_risk():
    data = {**_valid_data(), "geo_risk_tier": 6.5}
    errors = validate_claim_data(data)
    assert any("geo_risk_tier" in e for e in errors)


@pytest.mark.parametrize("claimed,limit,prior,expected", [
    (70_000, 100_000, 1, True),   # severity_ratio 0.7
    (40_000, 100_000, 5, True),   # high prior claims
    (10_000, 100_000, 1, False),  # normal
    (60_001, 100_000, 0, True),   # just above 0.6
])
def test_is_high_risk_claim(claimed, limit, prior, expected):
    assert is_high_risk_claim(claimed, limit, prior) == expected
