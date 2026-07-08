"""Drift detection and prediction logging for model monitoring.

Uses the Kolmogorov-Smirnov two-sample test to compare a reference window
of feature values against a recent current window. A p-value below
DRIFT_THRESHOLD (0.05) indicates statistically significant distribution shift.

Key functions
-------------
compute_drift   – pure statistic, no DB dependency
log_prediction  – write a prediction record to the DB
run_drift_check – compute drift and persist the report
"""

import json
import logging
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)

REFERENCE_WINDOW = 1000
DRIFT_THRESHOLD = 0.05


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_THRESHOLD),
        "n_reference": len(reference),
        "n_current": len(current),
    }


def log_prediction(
    db: Session,
    predicted_value: float,
    features: dict[str, Any],
    correlation_id: str,
    investment_score: float | None = None,
    model_version: str = "1.0.0",
    property_id: int | None = None,
) -> PredictionLog:
    record = PredictionLog(
        property_id=property_id,
        predicted_value=predicted_value,
        investment_score=investment_score,
        model_version=model_version,
        features_json=json.dumps(features),
        correlation_id=correlation_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug(
        "Logged prediction id=%s corr=%s value=%.0f", record.id, correlation_id, predicted_value
    )
    return record


def run_drift_check(
    db: Session,
    feature_name: str,
    reference_values: list[float],
    current_values: list[float],
) -> DriftReport:
    result = compute_drift(reference_values, current_values)
    report = DriftReport(
        feature_name=feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=result["drift_detected"],
        sample_size=len(current_values),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    if result["drift_detected"]:
        logger.warning(
            "Drift detected on feature=%s ks=%.4f p=%.4f",
            feature_name,
            result["ks_statistic"],
            result["p_value"],
        )
    return report


def get_recent_predictions(db: Session, limit: int = REFERENCE_WINDOW) -> list[PredictionLog]:
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()


def get_drift_summary(db: Session) -> list[dict[str, Any]]:
    reports = db.query(DriftReport).order_by(DriftReport.created_at.desc()).limit(100).all()
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
