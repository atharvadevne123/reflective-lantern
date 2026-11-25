"""Extended feature engineering tests with parametrize and boundary coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    ActuarialFeatureEngineer,
    build_feature_pipeline,
    get_feature_names,
    prepare_input,
)


def _base_row(**overrides) -> dict:
    row = {
        "claim_age_days": 30.0,
        "policy_limit": 100_000.0,
        "claimed_amount": 20_000.0,
        "num_prior_claims": 1.0,
        "policy_tenure_years": 3.0,
        "geo_risk_tier": 2.0,
        "days_since_last_claim": 200.0,
        "policy_type": "auto",
    }
    row.update(overrides)
    return row


def _make_df_from_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


@pytest.mark.parametrize("n", [1, 5, 50, 200])
def test_feature_engineer_various_sizes(n: int) -> None:
    rng = np.random.default_rng(n)
    df = pd.DataFrame({
        "claim_age_days": rng.integers(0, 365, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000], n).astype(float),
        "claimed_amount": rng.exponential(20_000, n),
        "num_prior_claims": rng.integers(0, 5, n).astype(float),
        "policy_tenure_years": rng.uniform(1, 10, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(30, 500, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out.shape[0] == n
    assert out.shape[1] == 12


@pytest.mark.parametrize("age_days,expected_tier", [
    (0.0, 0.0),
    (7.0, 0.0),
    (8.0, 1.0),
    (30.0, 1.0),
    (31.0, 2.0),
    (90.0, 2.0),
    (91.0, 3.0),
    (365.0, 3.0),
    (366.0, 4.0),
    (9000.0, 4.0),
])
def test_age_tier_boundary(age_days: float, expected_tier: float) -> None:
    df = _make_df_from_rows([_base_row(claim_age_days=age_days)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["age_tier"].iloc[0] == expected_tier


@pytest.mark.parametrize("prior_claims,flag", [
    (0.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (3.0, 1.0),
    (5.0, 1.0),
    (10.0, 1.0),
])
def test_high_frequency_flag_boundary(prior_claims: float, flag: float) -> None:
    df = _make_df_from_rows([_base_row(num_prior_claims=prior_claims)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["high_frequency_flag"].iloc[0] == flag


@pytest.mark.parametrize("days,flag", [
    (0.0, 1.0),
    (45.0, 1.0),
    (89.0, 1.0),
    (90.0, 0.0),
    (200.0, 0.0),
    (999.0, 0.0),
])
def test_recency_flag_boundary(days: float, flag: float) -> None:
    df = _make_df_from_rows([_base_row(days_since_last_claim=days)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["recency_flag"].iloc[0] == flag


@pytest.mark.parametrize("claimed,limit,expected_ratio", [
    (0.0, 100_000.0, pytest.approx(0.0, abs=1e-4)),
    (50_000.0, 100_000.0, pytest.approx(0.5, abs=1e-4)),
    (100_000.0, 100_000.0, pytest.approx(1.0, abs=1e-4)),
    (200_000.0, 100_000.0, pytest.approx(2.0, abs=1e-4)),
])
def test_severity_ratio_values(claimed: float, limit: float, expected_ratio) -> None:
    df = _make_df_from_rows([_base_row(claimed_amount=claimed, policy_limit=limit)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["severity_ratio"].iloc[0] == expected_ratio


def test_log_claimed_is_log1p() -> None:
    df = _make_df_from_rows([_base_row(claimed_amount=10_000.0)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["log_claimed"].iloc[0] == pytest.approx(np.log1p(10_000.0), rel=1e-6)


def test_log_policy_limit_is_log1p() -> None:
    df = _make_df_from_rows([_base_row(policy_limit=250_000.0)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["log_policy_limit"].iloc[0] == pytest.approx(np.log1p(250_000.0), rel=1e-6)


def test_prior_claim_rate_formula() -> None:
    df = _make_df_from_rows([_base_row(num_prior_claims=4.0, policy_tenure_years=2.0)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    expected = 4.0 / (2.0 + 1e-9)
    assert out["prior_claim_rate"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_geo_risk_passthrough() -> None:
    df = _make_df_from_rows([_base_row(geo_risk_tier=4.0)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["geo_risk"].iloc[0] == 4.0


@pytest.mark.parametrize("policy_type,expected_enc", [
    ("home", 0),
    ("auto", 1),
    ("life", 2),
    ("commercial", 3),
])
def test_policy_type_encoding_all_types(policy_type: str, expected_enc: int) -> None:
    df = _make_df_from_rows([_base_row(policy_type=policy_type)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["policy_type_enc"].iloc[0] == expected_enc


def test_unknown_policy_type_defaults_to_auto() -> None:
    df = _make_df_from_rows([_base_row(policy_type="unknown_type")])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["policy_type_enc"].iloc[0] == 1


def test_sev_age_interaction_zero_when_age_tier_zero() -> None:
    df = _make_df_from_rows([_base_row(claim_age_days=3.0)])
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["sev_age_interaction"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_pipeline_output_no_nans() -> None:
    rng = np.random.default_rng(99)
    n = 100
    df = pd.DataFrame({
        "claim_age_days": rng.integers(0, 365, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000], n).astype(float),
        "claimed_amount": rng.exponential(20_000, n),
        "num_prior_claims": rng.integers(0, 5, n).astype(float),
        "policy_tenure_years": rng.uniform(1, 10, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(30, 500, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(df)
    assert not np.isnan(result).any()


def test_get_feature_names_returns_list() -> None:
    names = get_feature_names()
    assert isinstance(names, list)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_prepare_input_single_row_shape() -> None:
    raw = _base_row()
    df = prepare_input(raw)
    assert df.shape == (1, len(raw))


def test_prepare_input_preserves_values() -> None:
    raw = _base_row(claimed_amount=55_000.0)
    df = prepare_input(raw)
    assert df["claimed_amount"].iloc[0] == 55_000.0
