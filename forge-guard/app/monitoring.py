"""Drift detection and prediction logging for Forge-Guard."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "temperature",
    "pressure",
    "vibration",
    "cycle_time",
    "tool_wear",
    "power_consumption",
    "humidity",
]

REFERENCE_STATS: dict[str, list[float]] = {}


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a two-sample KS test between reference and current feature distributions.

    Returns a dict with ks_statistic, p_value, and drift_detected flag.
    """
    if len(reference) < 5 or len(current) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def log_prediction(
    db: Session,
    sensor_data: dict[str, float],
    prediction: int,
    defect_probability: float,
    model_version: str = "1.0.0",
    correlation_id: str | None = None,
) -> PredictionLog:
    """Persist a single inference record to the prediction_logs table."""
    record = PredictionLog(
        temperature=sensor_data.get("temperature", 0.0),
        pressure=sensor_data.get("pressure", 0.0),
        vibration=sensor_data.get("vibration", 0.0),
        cycle_time=sensor_data.get("cycle_time", 0.0),
        tool_wear=sensor_data.get("tool_wear", 0.0),
        power_consumption=sensor_data.get("power_consumption", 0.0),
        humidity=sensor_data.get("humidity", 0.0),
        prediction=prediction,
        defect_probability=defect_probability,
        model_version=model_version,
        correlation_id=correlation_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug("Prediction logged: id=%d prob=%.4f", record.id, defect_probability)
    return record


def run_drift_check(
    db: Session,
    model_version: str = "1.0.0",
    window_hours: int = 24,
    window_size: int = 500,
) -> list[dict[str, Any]]:
    """Compare recent predictions against reference window using KS test.

    Persists a DriftReport row for each feature and returns the results.
    """
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    recent = (
        db.query(PredictionLog)
        .filter(PredictionLog.timestamp >= cutoff)
        .order_by(PredictionLog.timestamp.desc())
        .limit(window_size)
        .all()
    )

    reference_q = (
        db.query(PredictionLog)
        .filter(PredictionLog.timestamp < cutoff)
        .order_by(PredictionLog.timestamp.desc())
        .limit(window_size)
        .all()
    )

    if not recent or not reference_q:
        logger.info("Insufficient data for drift check.")
        return []

    results = []
    for feat in FEATURE_COLS:
        ref_vals = [getattr(r, feat) for r in reference_q]
        cur_vals = [getattr(r, feat) for r in recent]
        drift = compute_drift(ref_vals, cur_vals)

        report = DriftReport(
            feature=feat,
            ks_statistic=drift["ks_statistic"],
            p_value=drift["p_value"],
            drift_detected=drift["drift_detected"],
            window_size=len(cur_vals),
            model_version=model_version,
        )
        db.add(report)

        results.append({"feature": feat, **drift})
        if drift["drift_detected"]:
            logger.warning(
                "DRIFT DETECTED on '%s': KS=%.4f p=%.4f",
                feat,
                drift["ks_statistic"],
                drift["p_value"],
            )

    db.commit()
    return results


def defect_rate(db: Session, hours: int = 24) -> dict[str, Any]:
    """Return defect rate stats for the recent window."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.timestamp >= cutoff)
        .all()
    )
    if not logs:
        return {"total": 0, "defects": 0, "defect_rate": 0.0}
    total = len(logs)
    defects = sum(1 for r in logs if r.prediction == 1)
    return {
        "total": total,
        "defects": defects,
        "defect_rate": round(defects / total, 4),
    }
