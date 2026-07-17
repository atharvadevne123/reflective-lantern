"""Drift detection and prediction logging for Suite-Cast."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from .database import ModelMetrics, Prediction

logger = logging.getLogger(__name__)


def log_prediction(
    db: Session,
    input_features: dict[str, Any],
    demand_score: float,
    suggested_rate: float,
    model_version: str,
) -> str:
    """Persist a prediction record and return its request_id.

    Args:
        db: Active SQLAlchemy session.
        input_features: Raw booking request fields as a dict.
        demand_score: Model-predicted booking demand probability.
        suggested_rate: Dynamically priced room rate suggestion.
        model_version: Version tag of the model that produced this prediction.

    Returns:
        UUID string identifying this prediction record.
    """
    request_id = str(uuid.uuid4())
    pred = Prediction(
        request_id=request_id,
        timestamp=datetime.utcnow(),
        input_features=input_features,
        demand_score=demand_score,
        suggested_rate=suggested_rate,
        model_version=model_version,
    )
    db.add(pred)
    db.commit()
    logger.info(
        "Logged prediction request_id=%s demand=%.4f rate=%.2f",
        request_id,
        demand_score,
        suggested_rate,
    )
    return request_id


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Kolmogorov-Smirnov test for distribution drift between two score lists.

    Args:
        reference: Demand scores from the training/reference window.
        current: Demand scores from recent production predictions.

    Returns:
        Dict with ks_statistic, p_value, drift_detected flag, and
        an optional reason string when data is insufficient.
    """
    if len(reference) < 5 or len(current) < 5:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reason": "insufficient_data",
        }
    stat, p_value = ks_2samp(reference, current)
    drift_detected = bool(p_value < 0.05)
    if drift_detected:
        logger.warning("Drift detected: KS=%.4f p=%.4f — consider retraining", stat, p_value)
    return {
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": drift_detected,
    }


def get_prediction_stats(db: Session, limit: int = 200) -> dict[str, Any]:
    """Aggregate statistics for the most recent predictions.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of recent records to aggregate.

    Returns:
        Dict with count, avg/min/max demand_score, and avg suggested_rate.
    """
    preds = db.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit).all()
    if not preds:
        return {
            "count": 0,
            "avg_demand_score": None,
            "avg_suggested_rate": None,
            "min_demand_score": None,
            "max_demand_score": None,
        }

    scores = [p.demand_score for p in preds]
    rates = [p.suggested_rate for p in preds]
    return {
        "count": len(preds),
        "avg_demand_score": round(sum(scores) / len(scores), 4),
        "avg_suggested_rate": round(sum(rates) / len(rates), 2),
        "min_demand_score": round(min(scores), 4),
        "max_demand_score": round(max(scores), 4),
    }


def log_model_metrics(db: Session, model_version: str, metrics: dict[str, Any]) -> None:
    """Persist model training metrics to the database.

    Args:
        db: Active SQLAlchemy session.
        model_version: Version tag to associate with these metrics.
        metrics: Dict containing auc_mean, auc_std, n_features.
    """
    record = ModelMetrics(
        model_version=model_version,
        auc_mean=float(metrics.get("auc_mean", 0.0)),
        auc_std=float(metrics.get("auc_std", 0.0)),
        n_features=int(metrics.get("n_features", 0)),
        trained_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    logger.info(
        "Persisted model metrics version=%s AUC=%.4f",
        model_version,
        metrics.get("auc_mean", 0.0),
    )
