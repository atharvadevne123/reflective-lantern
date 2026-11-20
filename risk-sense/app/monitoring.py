"""Drift detection, prediction logging, and monitoring for Risk-Sense."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD_P = 0.05


def ks_drift_test(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a two-sample Kolmogorov-Smirnov test to detect distribution drift.

    Uses scipy.stats.ks_2samp; drift is flagged when p_value < DRIFT_THRESHOLD_P (0.05).
    Returns early with drift_detected=False when either sample has fewer than 10 points,
    because KS power is very low for tiny samples and would produce noisy false positives.
    """
    if len(reference) < 10 or len(current) < 10:
        return {"drift_detected": False, "reason": "insufficient_data", "ks_statistic": 0.0, "p_value": 1.0}

    stat, p_value = stats.ks_2samp(reference, current)
    drift_detected = bool(p_value < DRIFT_THRESHOLD_P)

    return {
        "drift_detected": drift_detected,
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "severity": drift_severity(stat),
    }


def drift_severity(ks_statistic: float) -> str:
    """Map KS statistic to severity label."""
    if ks_statistic < 0.1:
        return "none"
    if ks_statistic < 0.25:
        return "mild"
    if ks_statistic < 0.5:
        return "moderate"
    return "severe"


def compute_population_stability_index(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Compute the Population Stability Index (PSI) between two distributions.

    PSI < 0.1  → stable;  0.1–0.25 → moderate shift;  > 0.25 → major shift.
    Small epsilon (1e-7) added to each bin to avoid log(0).
    Joint bin edges (computed over both samples combined) ensure the same
    boundaries apply to both histograms, which is required for a valid PSI.
    """
    all_values = reference + current
    min_val, max_val = min(all_values), max(all_values)
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    ref_hist, _ = np.histogram(reference, bins=bin_edges)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_hist / max(len(reference), 1) + 1e-7
    cur_pct = cur_hist / max(len(current), 1) + 1e-7

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def rolling_default_rate(predictions: list[float], window: int = 100) -> list[float]:
    """Compute rolling mean default probability over a sliding window."""
    if not predictions:
        return []
    result = []
    for i in range(len(predictions)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(predictions[start : i + 1])))
    return result


def summarise_feature_drift(
    reference_data: dict[str, list[float]],
    current_data: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Run KS drift tests for all features and return summary report."""
    report = []
    for feature in reference_data:
        if feature not in current_data:
            continue
        result = ks_drift_test(reference_data[feature], current_data[feature])
        result["feature"] = feature
        report.append(result)
    return report


def log_prediction(
    db_session: Any,
    applicant_id: str,
    risk_score: float,
    risk_category: str,
    default_probability: float,
    features: dict[str, Any],
) -> None:
    """Persist a credit risk prediction to the database."""
    from app.database import CreditApplication

    record = CreditApplication(
        applicant_id=applicant_id,
        risk_score=risk_score,
        risk_category=risk_category,
        default_probability=default_probability,
        debt_to_income=features.get("debt_to_income"),
        payment_history_score=features.get("payment_history_score"),
        credit_utilization=features.get("credit_utilization"),
        account_age_months=features.get("account_age_months"),
        num_open_accounts=features.get("num_open_accounts"),
        num_derogatory_marks=features.get("num_derogatory_marks"),
        annual_income=features.get("annual_income"),
        created_at=datetime.utcnow(),
    )
    db_session.add(record)
    db_session.commit()
    logger.info("Logged prediction for applicant %s: score=%.4f", applicant_id, risk_score)
