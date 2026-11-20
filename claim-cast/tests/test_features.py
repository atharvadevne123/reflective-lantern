"""Feature engineering tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import ActuarialFeatureEngineer, build_feature_pipeline, prepare_input


def _make_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "claim_age_days": rng.integers(0, 365, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000], n).astype(float),
        "claimed_amount": rng.exponential(20_000, n),
        "num_prior_claims": rng.integers(0, 5, n).astype(float),
        "policy_tenure_years": rng.uniform(1, 10, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(30, 500, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })


def test_feature_engineer_output_shape():
    df = _make_df(10)
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == 10
    assert out.shape[1] == 12


def test_feature_engineer_no_nans():
    df = _make_df(50)
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert not out.isnull().any().any()


def test_severity_ratio_clipped():
    df = pd.DataFrame({
        "policy_limit": [100_000.0],
        "claimed_amount": [25_000.0],
        "claim_age_days": [10.0],
        "num_prior_claims": [0.0],
        "policy_tenure_years": [2.0],
        "geo_risk_tier": [2.0],
        "days_since_last_claim": [200.0],
        "policy_type": ["auto"],
    })
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["severity_ratio"].iloc[0] == pytest.approx(0.25, abs=1e-4)


def test_high_frequency_flag():
    df = _make_df(5)
    df["num_prior_claims"] = [0, 1, 2, 3, 5]
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert list(out["high_frequency_flag"]) == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_recency_flag():
    df = _make_df(3)
    df["days_since_last_claim"] = [30.0, 89.0, 100.0]
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert list(out["recency_flag"]) == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("policy_type,expected_enc", [
    ("home", 0), ("auto", 1), ("life", 2), ("commercial", 3),
])
def test_policy_type_encoding(policy_type, expected_enc):
    df = _make_df(1)
    df["policy_type"] = [policy_type]
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["policy_type_enc"].iloc[0] == expected_enc


def test_build_feature_pipeline_transforms():
    df = _make_df(20)
    pipe = build_feature_pipeline()
    result = pipe.fit_transform(df)
    assert result.shape[0] == 20


def test_prepare_input_returns_dataframe():
    raw = {
        "claim_age_days": 10.0,
        "policy_limit": 100_000.0,
        "claimed_amount": 20_000.0,
        "num_prior_claims": 1.0,
        "policy_tenure_years": 2.0,
        "geo_risk_tier": 3.0,
        "days_since_last_claim": 100.0,
        "policy_type": "home",
    }
    df = prepare_input(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


@pytest.mark.parametrize("age_days,expected_tier", [
    (3, 0.0),
    (15, 1.0),
    (60, 2.0),
    (200, 3.0),
    (500, 4.0),
])
def test_age_tier_values(age_days, expected_tier):
    df = _make_df(1)
    df["claim_age_days"] = [float(age_days)]
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    assert out["age_tier"].iloc[0] == expected_tier


def test_sev_age_interaction_matches():
    df = _make_df(1)
    df["claimed_amount"] = [30_000.0]
    df["policy_limit"] = [100_000.0]
    df["claim_age_days"] = [15.0]
    eng = ActuarialFeatureEngineer()
    out = eng.fit_transform(df)
    expected = out["severity_ratio"].iloc[0] * out["age_tier"].iloc[0]
    assert abs(out["sev_age_interaction"].iloc[0] - expected) < 1e-6
