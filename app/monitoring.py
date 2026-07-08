"""Model monitoring: KS-test drift detection and structured prediction logging."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

MODEL_VERSION = "1.0.0"

logger = logging.getLogger(__name__)


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a two-sample KS test to detect distribution shift between two windows."""
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reason": "insufficient_data",
        }
    stat, p_value = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": bool(p_value < 0.05),
    }


def log_prediction(
    db: Session,
    hour: int,
    day_of_week: int,
    route_id: str,
    road_type: str,
    prediction: dict[str, Any],
    features: dict[str, Any],
    model_version: str = MODEL_VERSION,
) -> PredictionLog:
    entry = PredictionLog(
        timestamp=datetime.utcnow(),
        hour=hour,
        day_of_week=day_of_week,
        route_id=route_id,
        road_type=road_type,
        congestion_level=prediction["congestion_level"],
        congestion_prob=prediction["congestion_probability"],
        incident_score=prediction["incident_score"],
        model_version=model_version,
        features_json=json.dumps(features),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info(
        "Logged prediction id=%d congestion_level=%d route=%s",
        entry.id,
        entry.congestion_level,
        route_id,
    )
    return entry


def log_drift(
    db: Session,
    feature_name: str,
    drift_result: dict[str, Any],
) -> DriftLog:
    entry = DriftLog(
        timestamp=datetime.utcnow(),
        feature_name=feature_name,
        ks_statistic=drift_result["ks_statistic"],
        p_value=drift_result["p_value"],
        drift_detected=int(drift_result["drift_detected"]),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    if drift_result["drift_detected"]:
        logger.warning(
            "DRIFT DETECTED feature=%s ks=%.4f p=%.4f",
            feature_name,
            drift_result["ks_statistic"],
            drift_result["p_value"],
        )
    return entry


def get_recent_predictions(db: Session, limit: int = 100) -> list[PredictionLog]:
    return db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()


def get_drift_summary(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    logs = db.query(DriftLog).order_by(DriftLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "feature": log.feature_name,
            "ks_statistic": log.ks_statistic,
            "p_value": log.p_value,
            "drift_detected": bool(log.drift_detected),
            "timestamp": log.timestamp.isoformat(),
        }
        for log in logs
    ]
