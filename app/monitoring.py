"""Model monitoring: KS-test drift detection and prediction logging."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import AnomalyLog, DriftLog, PredictionLog

logger = logging.getLogger(__name__)

# In-memory reference window (populated after first training)
_reference_window: list[float] = []
REFERENCE_WINDOW_SIZE = 500
DRIFT_P_THRESHOLD = 0.05


def set_reference_window(values: list[float]) -> None:
    """Set the reference distribution for drift detection."""
    global _reference_window
    _reference_window = list(values[-REFERENCE_WINDOW_SIZE:])
    logger.info("Reference window set: %d samples", len(_reference_window))


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS-test between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "reason": "insufficient_data"}
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
    building_id: str,
    timestamp: datetime,
    predicted_kwh: float,
    latency_ms: float,
    actual_kwh: float | None = None,
    model_version: str = "1.0.0",
) -> None:
    """Persist a prediction record; rolls back on DB error."""
    entry = PredictionLog(
        building_id=building_id,
        timestamp=timestamp,
        predicted_kwh=predicted_kwh,
        actual_kwh=actual_kwh,
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to log prediction for building %s", building_id)


def log_anomaly(
    db: Session,
    building_id: str,
    timestamp: datetime,
    consumption_kwh: float,
    anomaly_score: float,
    is_anomaly: int,
    severity: str,
) -> None:
    """Persist an anomaly detection record; rolls back on DB error."""
    entry = AnomalyLog(
        building_id=building_id,
        timestamp=timestamp,
        consumption_kwh=consumption_kwh,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly,
        severity=severity,
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to log anomaly for building %s", building_id)


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


def get_recent_predictions(db: Session, limit: int = 200) -> list[PredictionLog]:
    """Return the most recent prediction log entries."""
    return db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(limit).all()


def get_drift_summary(db: Session) -> list[dict[str, Any]]:
    """Return the latest drift detection result per feature."""
    from sqlalchemy import func

    subq = (
        db.query(DriftLog.feature_name, func.max(DriftLog.checked_at).label("latest"))
        .group_by(DriftLog.feature_name)
        .subquery()
    )
    rows = (
        db.query(DriftLog)
        .join(subq, (DriftLog.feature_name == subq.c.feature_name) & (DriftLog.checked_at == subq.c.latest))
        .all()
    )
    return [
        {
            "feature_name": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
            "checked_at": r.checked_at.isoformat() if r.checked_at else "",
        }
        for r in rows
    ]


def run_drift_check(db: Session, feature_name: str, reference: list[float], current: list[float]) -> Any:
    """Run a KS drift check and persist the result, returning a simple result object."""
    from types import SimpleNamespace

    result = compute_drift(reference, current)
    entry = DriftLog(
        feature_name=feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=int(result["drift_detected"]),
        checked_at=datetime.utcnow(),
    )
    db.add(entry)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist drift check for feature %s", feature_name)
    return SimpleNamespace(**result)


class LatencyTimer:
    """Context manager to measure request latency in milliseconds."""

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000, 2)
