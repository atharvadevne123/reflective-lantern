"""Drift detection and prediction logging using KS-test."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import ClaimPrediction, DriftLog

logger = logging.getLogger(__name__)


def log_prediction(
    db: Session,
    claim_id: str,
    severity_score: float,
    severity_label: str,
    triage_priority: int,
    model_version: str = "1.0.0",
) -> ClaimPrediction:
    record = ClaimPrediction(
        claim_id=claim_id,
        severity_score=severity_score,
        severity_label=severity_label,
        triage_priority=triage_priority,
        model_version=model_version,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("logged_prediction claim_id=%s severity=%s", claim_id, severity_label)
    return record


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "error": "insufficient_samples",
        }
    stat, p = ks_2samp(reference, current)
    drift_detected = p < 0.05
    logger.info("drift_check ks=%.4f p=%.4f drift=%s", stat, p, drift_detected)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(drift_detected),
    }


def log_drift(
    db: Session,
    feature_name: str,
    ks_statistic: float,
    p_value: float,
    drift_detected: bool,
    sample_size: int,
) -> DriftLog:
    record = DriftLog(
        feature_name=feature_name,
        ks_statistic=ks_statistic,
        p_value=p_value,
        drift_detected=int(drift_detected),
        sample_size=sample_size,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_recent_scores(db: Session, limit: int = 200) -> list[float]:
    rows = db.query(ClaimPrediction.severity_score).order_by(ClaimPrediction.id.desc()).limit(limit).all()
    return [r[0] for r in rows]
