"""Drift detection and prediction logging for Signal-Forge."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from .database import DriftEvent, MarketRegimePrediction

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.05
REFERENCE_WINDOW_DAYS = 30
MONITOR_WINDOW_DAYS = 7

FEATURE_NAMES = ["volatility", "momentum", "volume_ratio", "correlation", "beta"]


def log_prediction(
    db: Session,
    ticker: str,
    regime: str,
    confidence: float,
    risk_score: float,
    features: dict[str, float],
    model_version: str = "1.0.0",
) -> MarketRegimePrediction:
    """Persist a prediction record to the database."""
    import json
    record = MarketRegimePrediction(
        ticker=ticker,
        regime=regime,
        confidence=confidence,
        risk_score=risk_score,
        volatility=features.get("volatility"),
        momentum=features.get("momentum"),
        correlation=features.get("correlation"),
        volume_ratio=features.get("volume_ratio"),
        beta=features.get("beta"),
        features_json=json.dumps(features),
        model_version=model_version,
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info("Prediction logged: id=%d, ticker=%s, regime=%s", record.id, ticker, regime)
        return record
    except Exception as e:
        db.rollback()
        logger.error("Failed to log prediction: %s", e)
        raise


def run_ks_drift_detection(
    db: Session,
    feature_name: str,
    reference_values: list[float],
    current_values: list[float],
) -> dict[str, Any]:
    """Apply two-sample KS test to detect feature distribution drift."""
    if len(reference_values) < 10 or len(current_values) < 5:
        logger.warning("Insufficient data for KS test on feature: %s", feature_name)
        return {"drift_detected": False, "reason": "insufficient_data"}

    ks_stat, p_value = stats.ks_2samp(reference_values, current_values)
    drift_detected = p_value < DRIFT_THRESHOLD

    event = DriftEvent(
        feature_name=feature_name,
        ks_statistic=float(ks_stat),
        p_value=float(p_value),
        drift_detected=int(drift_detected),
        reference_mean=float(np.mean(reference_values)),
        current_mean=float(np.mean(current_values)),
    )
    try:
        db.add(event)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Failed to save drift event: %s", e)

    if drift_detected:
        logger.warning(
            "DRIFT DETECTED: feature=%s, ks=%.4f, p=%.4f",
            feature_name, ks_stat, p_value,
        )
    return {
        "feature": feature_name,
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": drift_detected,
        "reference_mean": round(float(np.mean(reference_values)), 4),
        "current_mean": round(float(np.mean(current_values)), 4),
    }


def fetch_feature_values(
    db: Session,
    feature_name: str,
    days: int,
) -> list[float]:
    """Fetch historical feature values from the predictions table."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    col_map = {
        "volatility": MarketRegimePrediction.volatility,
        "momentum": MarketRegimePrediction.momentum,
        "volume_ratio": MarketRegimePrediction.volume_ratio,
        "correlation": MarketRegimePrediction.correlation,
        "beta": MarketRegimePrediction.beta,
    }
    col = col_map.get(feature_name)
    if col is None:
        return []
    try:
        rows = (
            db.query(col)
            .filter(MarketRegimePrediction.created_at >= cutoff)
            .filter(col.isnot(None))
            .all()
        )
        return [float(r[0]) for r in rows]
    except Exception as e:
        logger.error("Failed to fetch feature values: %s", e)
        return []


def monitor_all_features(db: Session) -> list[dict[str, Any]]:
    """Run drift detection for all tracked features."""
    results = []
    for feature in FEATURE_NAMES:
        ref_values = fetch_feature_values(db, feature, REFERENCE_WINDOW_DAYS)
        cur_values = fetch_feature_values(db, feature, MONITOR_WINDOW_DAYS)
        result = run_ks_drift_detection(db, feature, ref_values, cur_values)
        results.append(result)
    return results


def get_drift_summary(db: Session) -> dict[str, Any]:
    """Return summary of recent drift events."""
    try:
        cutoff = datetime.utcnow() - timedelta(days=7)
        total = db.query(DriftEvent).filter(DriftEvent.detected_at >= cutoff).count()
        detected = (
            db.query(DriftEvent)
            .filter(DriftEvent.detected_at >= cutoff)
            .filter(DriftEvent.drift_detected == 1)
            .count()
        )
        return {"total_checks": total, "drift_events": detected, "window_days": 7}
    except Exception as e:
        logger.error("Failed to get drift summary: %s", e)
        return {"total_checks": 0, "drift_events": 0, "window_days": 7}
