"""Model monitoring: KS-test drift detection and prediction logging."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from app.database import AnomalyLog, DriftLog, PredictionLog

logger = logging.getLogger(__name__)

REFERENCE_WINDOW = 1000
DRIFT_THRESHOLD = 0.05
DRIFT_REPORT_LIMIT = 100
DEFAULT_MODEL_VERSION = "1.0.0"
MIN_DRIFT_SAMPLES = 2


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a KS two-sample test and return a drift summary dict.

    Args:
        reference: Historical baseline feature values (up to REFERENCE_WINDOW).
        current: Recent feature values to compare against the baseline.

    Returns:
        Dict with ``ks_statistic``, ``p_value``, ``drift_detected``,
        ``n_reference``, and ``n_current`` keys.
    """
    if len(reference) < MIN_DRIFT_SAMPLES or len(current) < MIN_DRIFT_SAMPLES:
        raise ValueError(
            f"Both reference and current must have at least {MIN_DRIFT_SAMPLES} samples"
        )
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_P_THRESHOLD),
    }


def check_feature_drift(feature_values: dict[str, list[float]], db: Session) -> list[dict[str, Any]]:
    """Check drift per feature and log results."""
    results = []
    for feature_name, current_vals in feature_values.items():
        ref = _reference_window if _reference_window else current_vals
        result = compute_drift(ref, current_vals)
        result["feature"] = feature_name

        entry = DriftLog(
            feature_name=feature_name,
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=int(result["drift_detected"]),
            checked_at=datetime.utcnow(),
        )
        db.add(entry)
        if result["drift_detected"]:
            logger.warning("Drift detected on feature '%s': KS=%.4f p=%.4f", feature_name, result["ks_statistic"], result["p_value"])
        results.append(result)

    db.commit()
    return results


def log_prediction(
    db: Session,
    predicted_value: float,
    features: dict[str, Any],
    correlation_id: str,
    investment_score: float | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    property_id: int | None = None,
) -> PredictionLog:
    """Persist a prediction record to the database.

    Args:
        db: Active SQLAlchemy session.
        predicted_value: Model output value in dollars.
        features: Raw feature dict used for the prediction.
        correlation_id: Request-scoped trace ID.
        investment_score: Optional investment scoring output.
        model_version: Identifier of the model that made the prediction.
        property_id: FK to the stored property row, if applicable.

    Returns:
        The newly created and refreshed ``PredictionLog`` ORM instance.
    """
    record = PredictionLog(
        property_id=property_id,
        predicted_value=predicted_value,
        investment_score=investment_score,
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(entry)
    db.commit()


def log_anomaly(
    db: Session,
    building_id: str,
    timestamp: datetime,
    consumption_kwh: float,
    anomaly_score: float,
    is_anomaly: int,
    severity: str,
) -> None:
    """Persist an anomaly detection record."""
    entry = AnomalyLog(
        building_id=building_id,
        timestamp=timestamp,
        consumption_kwh=consumption_kwh,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        severity=severity,
    )
    db.add(entry)
    db.commit()


def get_prediction_stats(db: Session) -> dict[str, Any]:
    """Aggregate prediction statistics for the /metrics endpoint."""
    total = db.query(PredictionLog).count()
    anomalies = db.query(AnomalyLog).filter(AnomalyLog.is_anomaly == 1).count()
    drifts = db.query(DriftLog).filter(DriftLog.drift_detected == 1).count()
    return {
        "total_predictions": total,
        "total_anomalies_flagged": anomalies,
        "total_drift_events": drifts,
        "reference_window_size": len(_reference_window),
    }


def get_drift_summary(db: Session) -> list[dict[str, Any]]:
    """Return the most recent drift reports as serialisable dicts."""
    reports = db.query(DriftReport).order_by(DriftReport.created_at.desc()).limit(DRIFT_REPORT_LIMIT).all()
    return [
        {
            "feature_name": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": r.drift_detected,
            "sample_size": r.sample_size,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]
