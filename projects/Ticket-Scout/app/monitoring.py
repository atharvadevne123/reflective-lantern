"""Prediction logging and drift detection."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

# In-memory reference window for drift (populated at startup from training data)
_reference_features: dict[str, list[float]] = {}


def set_reference_distribution(feature_name: str, values: list[float]) -> None:
    """Store a reference distribution for a feature.

    Args:
        feature_name: Feature identifier (e.g. 'text_length').
        values: Reference sample drawn from training data.
    """
    _reference_features[feature_name] = list(values)
    logger.debug("Reference set for feature '%s' (%d samples)", feature_name, len(values))


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS-test between reference and current distributions.

    Args:
        reference: Reference feature values (training distribution).
        current: Current window feature values.

    Returns:
        Dict with ks_statistic, p_value, drift_detected flag.
    """
    if len(reference) < 10 or len(current) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def check_all_drift(
    current_window: dict[str, list[float]],
    db: Session,
) -> list[dict[str, Any]]:
    """Check drift for all monitored features and persist results.

    Args:
        current_window: Dict mapping feature_name -> current values.
        db: Active SQLAlchemy session.

    Returns:
        List of drift result dicts, one per feature.
    """
    results = []
    for feature, current_vals in current_window.items():
        ref_vals = _reference_features.get(feature, [])
        result = compute_drift(ref_vals, current_vals)
        result["feature"] = feature

        log_entry = DriftLog(
            feature=feature,
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=int(result["drift_detected"]),
        )
        db.add(log_entry)

        if result["drift_detected"]:
            logger.warning(
                "Drift detected on feature '%s': KS=%.4f p=%.4f",
                feature, result["ks_statistic"], result["p_value"],
            )
        results.append(result)

    db.commit()
    return results


def log_prediction(
    db: Session,
    ticket_id: str,
    subject: str,
    body: str,
    priority: str,
    predicted_category: str,
    sla_breach_prob: float,
    resolution_hours_pred: float,
    confidence: float,
    model_version: str = "1.0.0",
) -> PredictionLog:
    """Persist a single prediction record to the database.

    Args:
        db: Active SQLAlchemy session.
        ticket_id: Unique ticket identifier.
        subject: Ticket subject text.
        body: Ticket body text (first 500 chars stored).
        priority: Reported priority level.
        predicted_category: Model-predicted category.
        sla_breach_prob: Probability of SLA breach.
        resolution_hours_pred: Predicted resolution time in hours.
        confidence: Category prediction confidence.
        model_version: Deployed model version tag.

    Returns:
        Persisted PredictionLog ORM object.
    """
    entry = PredictionLog(
        ticket_id=ticket_id or str(uuid.uuid4()),
        subject=subject,
        body_snippet=body[:500] if body else "",
        priority=priority,
        predicted_category=predicted_category,
        sla_breach_prob=round(sla_breach_prob, 4),
        resolution_hours_pred=round(resolution_hours_pred, 2),
        confidence=round(confidence, 4),
        model_version=model_version,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.debug("Prediction logged: ticket_id=%s category=%s", ticket_id, predicted_category)
    return entry


def get_recent_predictions(db: Session, hours: int = 24) -> list[PredictionLog]:
    """Fetch predictions from the last N hours.

    Args:
        db: Active SQLAlchemy session.
        hours: Look-back window in hours.

    Returns:
        List of PredictionLog ORM objects.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return db.query(PredictionLog).filter(PredictionLog.created_at >= cutoff).all()


def build_current_window(predictions: list[PredictionLog]) -> dict[str, list[float]]:
    """Extract numeric feature distributions from recent predictions.

    Args:
        predictions: Recent PredictionLog entries.

    Returns:
        Dict mapping feature name to list of observed values.
    """
    return {
        "text_length": [float(len(p.subject or "")) for p in predictions],
        "sla_breach_prob": [p.sla_breach_prob for p in predictions],
        "resolution_hours_pred": [p.resolution_hours_pred for p in predictions],
        "confidence": [p.confidence for p in predictions],
    }


def detect_volume_anomaly(predictions: list, baseline_hourly_rate: float = 10.0) -> dict:
    """Flag if the current prediction volume is anomalously high.

    Args:
        predictions: Recent prediction records.
        baseline_hourly_rate: Expected predictions per hour at normal load.

    Returns:
        Dict with is_anomaly flag and observed_rate.
    """
    if not predictions:
        return {"is_anomaly": False, "observed_rate": 0.0, "baseline": baseline_hourly_rate}
    from datetime import datetime
    times = [p.created_at for p in predictions if p.created_at]
    if len(times) < 2:
        return {"is_anomaly": False, "observed_rate": 0.0, "baseline": baseline_hourly_rate}
    span_hours = max(
        (max(times) - min(times)).total_seconds() / 3600, 0.017
    )  # min 1 minute
    rate = len(predictions) / span_hours
    threshold = baseline_hourly_rate * 5.0
    return {
        "is_anomaly": rate > threshold,
        "observed_rate": round(rate, 2),
        "baseline": baseline_hourly_rate,
        "threshold": threshold,
    }
