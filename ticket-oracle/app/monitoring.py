"""Drift detection and prediction logging for Ticket-Oracle.

Implements KS-test (feature distribution shift) and PSI (Population
Stability Index) checks, plus helper to log each prediction to the
SQLAlchemy PredictionLog table.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

PSI_BINS = 10
PSI_ALERT_THRESHOLD = 0.2
KS_ALPHA = 0.05


def compute_ks_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Compute KS-test between reference and current feature distributions.

    Args:
        reference: Baseline feature values (training window).
        current: Recent inference feature values.

    Returns:
        Dictionary with ks_statistic, p_value, and drift_detected flag.
    """
    if len(reference) < 5 or len(current) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "sample_size": len(current)}
    stat, p = ks_2samp(reference, current)
    result = {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < KS_ALPHA),
        "sample_size": len(current),
    }
    if result["drift_detected"]:
        logger.warning("ks_drift_detected", extra=result)
    return result


def compute_psi(reference: list[float], current: list[float], bins: int = PSI_BINS) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: stable; 0.1-0.2: moderate shift; > 0.2: significant shift.

    Args:
        reference: Baseline distribution values.
        current: Current distribution values.
        bins: Number of histogram bins.

    Returns:
        PSI value (non-negative float).
    """
    if len(reference) < 5 or len(current) < 5:
        return 0.0

    breakpoints = np.linspace(
        min(min(reference), min(current)),
        max(max(reference), max(current)),
        bins + 1,
    )

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = (ref_counts / len(reference)).clip(1e-8)
    cur_pct = (cur_counts / len(current)).clip(1e-8)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def run_drift_check(
    db: Session,
    feature_name: str,
    reference: list[float],
    current: list[float],
) -> dict[str, Any]:
    """Run KS + PSI drift check and persist result to DriftLog.

    Args:
        db: SQLAlchemy session.
        feature_name: Human-readable name of the feature being checked.
        reference: Baseline feature values.
        current: Recent feature values.

    Returns:
        Combined drift result dictionary.
    """
    ks = compute_ks_drift(reference, current)
    psi = compute_psi(reference, current)

    row = DriftLog(
        feature_name=feature_name,
        ks_statistic=ks["ks_statistic"],
        p_value=ks["p_value"],
        psi_value=psi,
        drift_detected=ks["drift_detected"] or psi >= PSI_ALERT_THRESHOLD,
        sample_size=ks["sample_size"],
    )
    db.add(row)
    db.commit()

    result = {**ks, "psi_value": psi, "feature_name": feature_name}
    logger.info("drift_check_complete", extra=result)
    return result


def log_prediction(
    db: Session,
    ticket_id: str,
    prediction: dict[str, Any],
    model_version: str = "1.0.0",
) -> PredictionLog:
    """Persist a prediction result to the PredictionLog table.

    Args:
        db: SQLAlchemy session.
        ticket_id: Unique identifier for the ticket.
        prediction: Output dictionary from model.predict().
        model_version: Model semantic version string.

    Returns:
        The committed PredictionLog ORM object.
    """
    probs = prediction.get("priority_probabilities", {})
    row = PredictionLog(
        ticket_id=ticket_id,
        priority_predicted=prediction["priority"],
        priority_p1_prob=float(probs.get("P1", 0.0)),
        priority_p2_prob=float(probs.get("P2", 0.0)),
        priority_p3_prob=float(probs.get("P3", 0.0)),
        priority_p4_prob=float(probs.get("P4", 0.0)),
        sla_breach_risk=float(prediction["sla_breach_risk"]),
        sla_breach_predicted=bool(prediction["sla_breach_predicted"]),
        estimated_resolution_hours=float(prediction["estimated_resolution_hours"]),
        confidence=float(prediction["confidence"]),
        model_version=model_version,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("prediction_logged", extra={"ticket_id": ticket_id, "priority": prediction["priority"]})
    return row


def get_recent_predictions(db: Session, n: int = 500) -> list[PredictionLog]:
    """Retrieve the most recent n predictions for drift analysis.

    Args:
        db: SQLAlchemy session.
        n: Maximum number of records to return.

    Returns:
        List of PredictionLog ORM objects ordered by creation time descending.
    """
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(n).all()
