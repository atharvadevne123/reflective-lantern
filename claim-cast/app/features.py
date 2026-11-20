"""Actuarial feature engineering pipeline for insurance claims."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ActuarialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives actuarial features from raw claim attributes."""

    def fit(self, X: pd.DataFrame, y=None) -> ActuarialFeatureEngineer:
        self.feature_names_: list[str] = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Claim age tiers: days since incident to report
        df["claim_age_days"] = df.get("claim_age_days", pd.Series(np.zeros(len(df))))
        df["age_tier"] = pd.cut(
            df["claim_age_days"],
            bins=[-1, 7, 30, 90, 365, 9999],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        # Severity ratio: claimed amount vs policy limit
        policy_limit = df.get("policy_limit", pd.Series(np.ones(len(df)) * 100_000))
        claimed = df.get("claimed_amount", pd.Series(np.zeros(len(df))))
        df["severity_ratio"] = claimed / (policy_limit + 1e-9)
        df["log_claimed"] = np.log1p(claimed)
        df["log_policy_limit"] = np.log1p(policy_limit)

        # Rolling claim frequency proxy (num_prior_claims rolling stat)
        prior = df.get("num_prior_claims", pd.Series(np.zeros(len(df))))
        df["prior_claim_rate"] = prior / (df.get("policy_tenure_years", pd.Series(np.ones(len(df)))) + 1e-9)
        df["high_frequency_flag"] = (prior >= 3).astype(float)

        # Geographic risk tier (1–5 encoded)
        df["geo_risk"] = df.get("geo_risk_tier", pd.Series(np.ones(len(df)) * 2.5))

        # Lag features: days between prior claims
        df["days_since_last_claim"] = df.get("days_since_last_claim", pd.Series(np.full(len(df), 365.0)))
        df["recency_flag"] = (df["days_since_last_claim"] < 90).astype(float)

        # Interaction: severity ratio × age tier
        df["sev_age_interaction"] = df["severity_ratio"] * df["age_tier"]

        # Policy type encoding (home=0, auto=1, life=2, commercial=3)
        policy_map = {"home": 0, "auto": 1, "life": 2, "commercial": 3}
        df["policy_type_enc"] = df.get("policy_type", pd.Series(["auto"] * len(df))).map(policy_map).fillna(1)

        return df[self._get_feature_cols(df)]

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> list[str]:
        return [
            "claim_age_days", "age_tier", "severity_ratio", "log_claimed",
            "log_policy_limit", "prior_claim_rate", "high_frequency_flag",
            "geo_risk", "days_since_last_claim", "recency_flag",
            "sev_age_interaction", "policy_type_enc",
        ]


def build_feature_pipeline() -> Pipeline:
    return Pipeline([
        ("features", ActuarialFeatureEngineer()),
        ("scaler", StandardScaler()),
    ])


def prepare_input(raw: dict) -> pd.DataFrame:
    return pd.DataFrame([raw])


def get_feature_names() -> list[str]:
    """Return the ordered list of engineered feature names.

    Returns:
        List of feature name strings in pipeline output order.
    """
    return [
        "claim_age_days", "age_tier", "severity_ratio", "log_claimed",
        "log_policy_limit", "prior_claim_rate", "high_frequency_flag",
        "geo_risk", "days_since_last_claim", "recency_flag",
        "sev_age_interaction", "policy_type_enc",
    ]
