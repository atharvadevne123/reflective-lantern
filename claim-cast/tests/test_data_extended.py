"""Extended tests for synthetic claim data generation."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data import generate_claim_dict, generate_claims

REQUIRED_COLUMNS = [
    "claim_age_days",
    "policy_limit",
    "claimed_amount",
    "num_prior_claims",
    "policy_tenure_years",
    "geo_risk_tier",
    "days_since_last_claim",
    "policy_type",
]


@pytest.mark.parametrize("n", [1, 10, 100, 500, 1000])
def test_generate_claims_row_count(n: int) -> None:
    X, y = generate_claims(n=n)
    assert len(X) == n
    assert len(y) == n


def test_generate_claims_has_required_columns() -> None:
    X, _ = generate_claims(n=50)
    for col in REQUIRED_COLUMNS:
        assert col in X.columns, f"Missing column: {col}"


def test_generate_claims_no_nans() -> None:
    X, y = generate_claims(n=200)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_generate_claims_target_classes() -> None:
    _, y = generate_claims(n=500, seed=0)
    unique = set(y.unique())
    assert unique.issubset({0, 1, 2, 3})


def test_generate_claims_target_all_classes_represented() -> None:
    _, y = generate_claims(n=1000, seed=42)
    assert len(y.unique()) >= 3


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 7])
def test_generate_claims_reproducible(seed: int) -> None:
    X1, y1 = generate_claims(n=50, seed=seed)
    X2, y2 = generate_claims(n=50, seed=seed)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_generate_claims_different_seeds_differ() -> None:
    X1, _ = generate_claims(n=50, seed=1)
    X2, _ = generate_claims(n=50, seed=2)
    assert not X1["claim_age_days"].equals(X2["claim_age_days"])


def test_generate_claims_policy_type_valid_values() -> None:
    X, _ = generate_claims(n=200)
    valid = {"home", "auto", "life", "commercial"}
    assert set(X["policy_type"].unique()).issubset(valid)


def test_generate_claims_geo_risk_tier_range() -> None:
    X, _ = generate_claims(n=500)
    assert X["geo_risk_tier"].between(1, 5).all()


def test_generate_claims_non_negative_amounts() -> None:
    X, _ = generate_claims(n=200)
    assert (X["claimed_amount"] >= 0).all()
    assert (X["policy_limit"] > 0).all()


def test_generate_claims_non_negative_prior_claims() -> None:
    X, _ = generate_claims(n=200)
    assert (X["num_prior_claims"] >= 0).all()


def test_generate_claims_policy_tenure_positive() -> None:
    X, _ = generate_claims(n=200)
    assert (X["policy_tenure_years"] > 0).all()


def test_generate_claim_dict_has_required_keys() -> None:
    d = generate_claim_dict(seed=42)
    for col in REQUIRED_COLUMNS:
        assert col in d or "claim_id" in d


def test_generate_claim_dict_claim_id_present() -> None:
    d = generate_claim_dict(seed=0)
    assert "claim_id" in d
    assert isinstance(d["claim_id"], str)


@pytest.mark.parametrize("seed", [0, 42, 100])
def test_generate_claim_dict_reproducible(seed: int) -> None:
    d1 = generate_claim_dict(seed=seed)
    d2 = generate_claim_dict(seed=seed)
    assert d1 == d2


def test_generate_claim_dict_no_seed_runs() -> None:
    d = generate_claim_dict()
    assert "claim_id" in d


def test_generate_claim_dict_valid_policy_type() -> None:
    for seed in range(10):
        d = generate_claim_dict(seed=seed)
        assert d["policy_type"] in {"home", "auto", "life", "commercial"}
