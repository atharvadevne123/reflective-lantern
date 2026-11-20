"""Synthetic claim data generation utilities for testing and seeding."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_claims(n: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic insurance claims dataset.

    Args:
        n: Number of claim records to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (feature DataFrame, target severity Series).
    """
    rng = np.random.default_rng(seed)

    X = pd.DataFrame({
        "claim_age_days": rng.integers(0, 400, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000, 500_000], n).astype(float),
        "claimed_amount": rng.exponential(25_000, n),
        "num_prior_claims": rng.integers(0, 8, n).astype(float),
        "policy_tenure_years": rng.uniform(0.5, 20, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(10, 1000, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })

    ratio = (X["claimed_amount"] / X["policy_limit"]).clip(0, 1.0)
    y = pd.cut(ratio, bins=[-0.001, 0.1, 0.3, 0.6, 1.001], labels=[0, 1, 2, 3]).astype(int)

    return X, y


def generate_claim_dict(seed: int | None = None) -> dict:
    """Generate a single random claim as a dictionary.

    Args:
        seed: Optional random seed.

    Returns:
        Dict suitable for ClaimRequest.
    """
    rng = np.random.default_rng(seed)
    return {
        "claim_id": f"CLM-GEN-{rng.integers(1000, 9999)}",
        "claim_age_days": float(rng.integers(0, 400)),
        "policy_limit": float(rng.choice([50_000, 100_000, 250_000])),
        "claimed_amount": float(rng.exponential(25_000)),
        "num_prior_claims": float(rng.integers(0, 5)),
        "policy_tenure_years": float(rng.uniform(0.5, 15)),
        "geo_risk_tier": float(rng.integers(1, 6)),
        "days_since_last_claim": float(rng.integers(30, 500)),
        "policy_type": str(rng.choice(["home", "auto", "life", "commercial"])),
    }
