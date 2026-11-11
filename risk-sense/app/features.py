"""Feature engineering pipeline for credit risk scoring."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "debt_to_income",
    "payment_history_score",
    "credit_utilization",
    "account_age_months",
    "num_open_accounts",
    "num_derogatory_marks",
    "annual_income_log",
    "utilization_x_derogatory",
    "income_per_account",
]


class LogIncomeTransformer(BaseEstimator, TransformerMixin):
    """Log-transform annual income to reduce skew."""

    def fit(self, X: Any, y: Any = None) -> "LogIncomeTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = X.copy()
        result[:, 6] = np.log1p(np.clip(result[:, 6], 0, None))
        return result


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Adds interaction features: utilization*derogatory, income/accounts."""

    def fit(self, X: Any, y: Any = None) -> "InteractionFeatureTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = X.copy()
        utilization = result[:, 2]
        derogatory = result[:, 5]
        income_log = result[:, 6]
        accounts = np.clip(result[:, 4], 1, None)
        interaction = utilization * derogatory
        income_per_acct = income_log / accounts
        return np.column_stack([result, interaction, income_per_acct])


class ClipOutlierTransformer(BaseEstimator, TransformerMixin):
    """Clip extreme values to percentile bounds fitted on training data."""

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "ClipOutlierTransformer":
        self.lower_ = np.percentile(X, self.lower_pct, axis=0)
        self.upper_ = np.percentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower_, self.upper_)


def build_pipeline() -> Pipeline:
    """Return the full sklearn feature engineering + scaling pipeline."""
    return Pipeline([
        ("log_income", LogIncomeTransformer()),
        ("interactions", InteractionFeatureTransformer()),
        ("clip_outliers", ClipOutlierTransformer()),
        ("scaler", StandardScaler()),
    ])


def extract_raw_features(application: dict[str, Any]) -> np.ndarray:
    """Extract ordered feature array from application dict."""
    required = [
        "debt_to_income",
        "payment_history_score",
        "credit_utilization",
        "account_age_months",
        "num_open_accounts",
        "num_derogatory_marks",
        "annual_income",
    ]
    missing = [k for k in required if k not in application]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    return np.array([[
        float(application["debt_to_income"]),
        float(application["payment_history_score"]),
        float(application["credit_utilization"]),
        float(application["account_age_months"]),
        float(application["num_open_accounts"]),
        float(application["num_derogatory_marks"]),
        float(application["annual_income"]),
    ]])


def risk_category(score: float) -> str:
    """Map probability score to risk category label."""
    if score < 0.2:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 0.75:
        return "high"
    return "very_high"


def payment_score_to_months(score: float) -> int:
    """Approximate account age in months from normalised payment score."""
    return max(1, math.floor(score * 120))


def feature_importance_report(features: list[str], importances: list[float]) -> list[dict[str, Any]]:
    """Return sorted feature importance report as list of dicts."""
    if len(features) != len(importances):
        raise ValueError("features and importances must have equal length")
    paired = sorted(zip(importances, features), reverse=True)
    return [{"feature": f, "importance": round(imp, 6)} for imp, f in paired]
