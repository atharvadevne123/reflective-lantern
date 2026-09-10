"""Drift detection and prediction logging for Cyber-Guard."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

_settings = get_settings()

REFERENCE_WINDOW_DAYS = _settings.reference_window_days
DRIFT_P_THRESHOLD = _settings.drift_p_threshold


def log_prediction(
    db: Session,
    features: dict[str, Any],
    prediction: str,
    confidence: float,
    model_version: str = "1.0.0",
) -> PredictionLog:
    """Persist a single prediction for later drift and volume analysis.

    Args:
        db: Active database session.
        features: Raw connection fields as submitted to the API.
        prediction: The predicted threat class.
        confidence: Probability assigned to the predicted class.
        model_version: Version string of the serving model.

    Returns:
        The committed PredictionLog row, with its id populated.
    """
    record = PredictionLog(
        src_bytes=float(features.get("src_bytes", 0)),
        dst_bytes=float(features.get("dst_bytes", 0)),
        duration=float(features.get("duration", 0)),
        protocol_type=str(features.get("protocol_type", "tcp")),
        service=str(features.get("service", "other")),
        flag=str(features.get("flag", "SF")),
        prediction=prediction,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        "prediction logged id=%d label=%s confidence=%.4f", record.id, prediction, confidence
    )
    return record


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test between two distributions.

    Args:
        reference: Values from the historical reference window.
        current: Values from the recent window.

    Returns:
        A dict with ``ks_statistic``, ``p_value`` and ``drift_detected``.
        When either sample has fewer than two points the test is not
        computable and an ``error`` key is returned instead of raising.
    """
    if len(reference) < 2 or len(current) < 2:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "error": "insufficient data",
        }
    stat, p = ks_2samp(reference, current)
    drift = bool(p < DRIFT_P_THRESHOLD)
    logger.info("drift check ks=%.4f p=%.4f drift=%s", stat, p, drift)
    # scipy returns numpy scalars; cast to native floats so the DB write and
    # any JSON consumer get predictable Python types.
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": drift,
    }


def run_drift_check(db: Session) -> dict[str, Any]:
    """Compare recent src_bytes against the reference window and record it.

    Args:
        db: Active database session.

    Returns:
        The drift result from :func:`compute_drift`.
    """
    cutoff = datetime.utcnow() - timedelta(days=REFERENCE_WINDOW_DAYS)
    recent = datetime.utcnow() - timedelta(hours=24)

    ref_rows = (
        db.query(PredictionLog.src_bytes).filter(PredictionLog.timestamp < cutoff).limit(500).all()
    )
    cur_rows = (
        db.query(PredictionLog.src_bytes).filter(PredictionLog.timestamp >= recent).limit(500).all()
    )

    ref_vals = [r[0] for r in ref_rows]
    cur_vals = [r[0] for r in cur_rows]
    result = compute_drift(ref_vals, cur_vals)

    drift_record = DriftLog(
        feature="src_bytes",
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=int(result["drift_detected"]),
    )
    db.add(drift_record)
    db.commit()

    return result


def get_prediction_stats(db: Session, hours: int = 24) -> dict[str, Any]:
    """Summarise predictions logged over a recent time window.

    Args:
        db: Active database session.
        hours: Size of the lookback window in hours.

    Returns:
        A dict with ``total``, ``hours``, ``class_counts`` and, when at least
        one row matched, ``avg_confidence``.
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = db.query(PredictionLog).filter(PredictionLog.timestamp >= since).all()

    if not rows:
        return {"total": 0, "hours": hours, "class_counts": {}}

    class_counts: dict[str, int] = {}
    for r in rows:
        class_counts[r.prediction] = class_counts.get(r.prediction, 0) + 1

    avg_confidence = sum(r.confidence for r in rows) / len(rows)

    return {
        "total": len(rows),
        "hours": hours,
        "avg_confidence": round(avg_confidence, 4),
        "class_counts": class_counts,
    }
