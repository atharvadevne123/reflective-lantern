"""Lightweight model explainability helpers for credit risk decisions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

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

RISK_FACTORS = {
    "debt_to_income": "High debt-to-income ratio increases default risk.",
    "credit_utilization": "High credit utilization suggests financial stress.",
    "num_derogatory_marks": "Past derogatory marks indicate credit mismanagement.",
    "payment_history_score": "Poor payment history is a strong default predictor.",
    "utilization_x_derogatory": "Combined high utilization and derogatory marks amplify risk.",
}


def top_risk_drivers(
    feature_values: np.ndarray,
    feature_importances: np.ndarray,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Return the top N features driving the risk score for a single applicant.

    Uses a simple contribution score: importance × |normalised_value|.
    This is intentionally lightweight — not SHAP — to avoid adding heavy
    dependencies to the inference path.
    """
    if feature_values.ndim == 2:
        feature_values = feature_values[0]

    if len(feature_values) != len(feature_importances):
        raise ValueError("feature_values and feature_importances must have equal length")

    contributions = np.abs(feature_values) * feature_importances
    ranked_idx = np.argsort(contributions)[::-1][:top_n]

    result = []
    for idx in ranked_idx:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        result.append(
            {
                "feature": name,
                "contribution": float(round(contributions[idx], 6)),
                "explanation": RISK_FACTORS.get(name, ""),
            }
        )
    return result


def adverse_action_reasons(
    feature_values: np.ndarray,
    feature_importances: np.ndarray,
    risk_score: float,
    threshold: float = 0.5,
) -> list[str]:
    """Return plain-language adverse action reasons when risk_score >= threshold.

    Required for regulatory compliance (ECOA/FCRA): lenders must state the
    principal reasons for an adverse action.
    """
    if risk_score < threshold:
        return []

    drivers = top_risk_drivers(feature_values, feature_importances, top_n=4)
    reasons = []
    for driver in drivers:
        explanation = driver.get("explanation")
        if explanation:
            reasons.append(explanation)
    return reasons[:4]


def score_band_description(risk_score: float) -> str:
    """Return a human-readable description of the risk band for the score."""
    if risk_score < 0.2:
        return "Low risk: strong credit profile with minimal default indicators."
    if risk_score < 0.5:
        return "Medium risk: some credit concerns; manual review recommended."
    if risk_score < 0.75:
        return "High risk: significant default indicators present."
    return "Very high risk: multiple severe default predictors detected."
