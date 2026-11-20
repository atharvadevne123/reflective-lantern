"""Prediction logging and KS-drift detection for News-Guard."""

import logging
from datetime import datetime, timedelta

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)


def compute_drift(reference: list[float], current: list[float]) -> dict:
    """KS-test between reference and current prediction score distributions."""
    if len(current) < 10:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "note": "insufficient_samples"}
    stat, p = ks_2samp(reference, current)
    drift = bool(p < 0.05)
    if drift:
        logger.warning("Drift detected: KS=%.4f p=%.4f", stat, p)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": drift,
    }


def get_recent_predictions(db: Session, hours: int = 24) -> list[float]:
    """Fetch fake_probability scores from recent predictions."""
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(PredictionLog.confidence)
        .filter(PredictionLog.created_at >= since)
        .filter(PredictionLog.prediction == "FAKE")
        .all()
    )
    return [float(r[0]) for r in rows]


def log_drift_check(
    db: Session,
    feature_name: str,
    ks_statistic: float,
    p_value: float,
    drift_detected: bool,
) -> DriftLog:
    record = DriftLog(
        feature_name=feature_name,
        ks_statistic=ks_statistic,
        p_value=p_value,
        drift_detected=int(drift_detected),
        checked_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_monitoring_summary(db: Session) -> dict:
    """Aggregate prediction stats for the /metrics endpoint."""
    total = db.query(PredictionLog).count()
    fake_count = db.query(PredictionLog).filter(PredictionLog.prediction == "FAKE").count()
    real_count = total - fake_count
    recent_drift = (
        db.query(DriftLog)
        .order_by(DriftLog.checked_at.desc())
        .first()
    )
    return {
        "total_predictions": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_rate": round(fake_count / max(total, 1), 4),
        "drift_detected": bool(recent_drift and recent_drift.drift_detected),
        "last_drift_check": recent_drift.checked_at.isoformat() if recent_drift else None,
    }
