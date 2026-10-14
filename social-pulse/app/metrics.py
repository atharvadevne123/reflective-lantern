"""Aggregated prediction metrics helpers for Social-Pulse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import DriftRecord, Prediction

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Snapshot of aggregated inference statistics.

    Attributes:
        total_predictions: Total number of predictions logged.
        viral_count: Count of predictions labelled viral.
        viral_rate: Fraction of predictions labelled viral.
        avg_confidence: Mean ensemble confidence (0–1).
        avg_latency_ms: Mean inference latency in milliseconds.
        model_version: Latest model version seen in the log.
        drifted_features: Features with active drift flags.
    """

    total_predictions: int
    viral_count: int
    viral_rate: float
    avg_confidence: float
    avg_latency_ms: float
    model_version: str
    drifted_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "total_predictions": self.total_predictions,
            "viral_count": self.viral_count,
            "viral_rate": round(self.viral_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "model_version": self.model_version,
            "drifted_features": self.drifted_features,
        }


def compute_metrics(db: Session, model_version: str) -> PredictionMetrics:
    """Compute aggregated metrics from the predictions table.

    Args:
        db: Active SQLAlchemy session.
        model_version: Current serving model version (used as fallback).

    Returns:
        A populated PredictionMetrics instance.
    """
    try:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        viral_count = (
            db.query(func.count(Prediction.id))
            .filter(Prediction.viral_label.is_(True))
            .scalar() or 0
        )
        avg_conf = float(db.query(func.avg(Prediction.ensemble_confidence)).scalar() or 0.0)
        avg_lat = float(db.query(func.avg(Prediction.latency_ms)).scalar() or 0.0)
        latest_version = (
            db.query(Prediction.model_version)
            .order_by(Prediction.created_at.desc())
            .limit(1)
            .scalar()
        ) or model_version

        # Active drifted features (most recent check per feature)
        drifted: list[str] = []
        subq = (
            db.query(
                DriftRecord.feature_name,
                func.max(DriftRecord.checked_at).label("latest"),
            )
            .group_by(DriftRecord.feature_name)
            .subquery()
        )
        drift_rows = (
            db.query(DriftRecord)
            .join(subq, (DriftRecord.feature_name == subq.c.feature_name) & (DriftRecord.checked_at == subq.c.latest))
            .filter(DriftRecord.drift_detected.is_(True))
            .all()
        )
        drifted = [r.feature_name for r in drift_rows]

        return PredictionMetrics(
            total_predictions=total,
            viral_count=viral_count,
            viral_rate=viral_count / total if total else 0.0,
            avg_confidence=avg_conf,
            avg_latency_ms=avg_lat,
            model_version=latest_version,
            drifted_features=drifted,
        )
    except Exception as exc:
        logger.error("compute_metrics failed: %s", exc)
        return PredictionMetrics(
            total_predictions=0,
            viral_count=0,
            viral_rate=0.0,
            avg_confidence=0.0,
            avg_latency_ms=0.0,
            model_version=model_version,
            drifted_features=[],
        )
