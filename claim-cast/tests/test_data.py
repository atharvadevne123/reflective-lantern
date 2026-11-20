"""Tests for synthetic data generation utilities."""
from __future__ import annotations

import pytest

from app.data import generate_claim_dict, generate_claims


def test_generate_claims_shape():
    X, y = generate_claims(n=100, seed=1)
    assert len(X) == 100
    assert len(y) == 100
    assert X.shape[1] == 8


def test_generate_claims_target_classes():
    _, y = generate_claims(n=200, seed=2)
    unique_labels = set(y.unique())
    assert unique_labels.issubset({0, 1, 2, 3})


def test_generate_claims_no_nulls():
    X, y = generate_claims(n=50, seed=3)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_generate_claims_reproducible():
    X1, y1 = generate_claims(n=20, seed=99)
    X2, y2 = generate_claims(n=20, seed=99)
    assert (X1 == X2).all().all()
    assert (y1 == y2).all()


def test_generate_claim_dict_has_required_keys():
    claim = generate_claim_dict(seed=7)
    required = {"claim_id", "claim_age_days", "policy_limit", "claimed_amount",
                "num_prior_claims", "policy_tenure_years", "geo_risk_tier",
                "days_since_last_claim", "policy_type"}
    assert required.issubset(claim.keys())


@pytest.mark.parametrize("seed", [1, 42, 100])
def test_generate_claim_dict_valid_policy_type(seed):
    claim = generate_claim_dict(seed=seed)
    assert claim["policy_type"] in {"home", "auto", "life", "commercial"}
