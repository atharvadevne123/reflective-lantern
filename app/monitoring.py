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
            logger.warning(
                "Drift detected on feature '%s': KS=%.4f p=%.4f",
                feature_name,
                result["ks_statistic"],
                result["p_value"],
            )
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


def reset_reference_window() -> None:
    """Clear the in-memory reference window (useful in tests)."""
    global _reference_window
    _reference_window = []


class LatencyTimer:
    """Context manager to measure request latency in milliseconds."""

    def __enter__(self) -> LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000, 2)


def get_anomaly_stats(db: Session) -> dict[str, Any]:
    """Return aggregated anomaly statistics from the database.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with total_anomalies, critical_count, warning_count, anomaly_rate,
        and mean_anomaly_score.
    """
    import statistics

    all_anomalies = db.query(AnomalyLog).all()
    total = len(all_anomalies)
    if total == 0:
        return {
            "total_anomalies": 0,
            "critical_count": 0,
            "warning_count": 0,
            "anomaly_rate": 0.0,
            "mean_anomaly_score": 0.0,
        }
    flagged = [a for a in all_anomalies if a.is_anomaly == 1]
    critical = [a for a in flagged if a.severity == "critical"]
    warning = [a for a in flagged if a.severity == "warning"]
    scores = [a.anomaly_score for a in all_anomalies]
    mean_score = round(statistics.mean(scores), 4) if scores else 0.0
    return {
        "total_anomalies": total,
        "critical_count": len(critical),
        "warning_count": len(warning),
        "anomaly_rate": round(len(flagged) / total, 4),
        "mean_anomaly_score": mean_score,
    }


def compute_feature_drift_summary(
    feature_values: dict[str, list[float]],
    reference: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute drift for multiple features without persisting to DB.

    Args:
        feature_values: Mapping of feature name to list of current values.
        reference: Shared reference distribution; falls back to _reference_window.

    Returns:
        List of drift result dicts, one per feature, with 'feature', 'ks_statistic',
        'p_value', and 'drift_detected' keys.
    """
    ref = reference if reference is not None else _reference_window
    results = []
    for name, current in feature_values.items():
        result = compute_drift(ref if ref else current, current)
        result["feature"] = name
        results.append(result)
    return results


def summarize_drift_history(drift_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of drift-check results into aggregate statistics.

    Args:
        drift_results: List of dicts, each with at least 'drift_detected',
            'ks_statistic', and 'p_value' keys (e.g. from compute_drift calls).

    Returns:
        Dict with 'total_checks', 'drift_count', 'drift_rate',
        'mean_ks_statistic', and 'min_p_value'.
    """
    if not drift_results:
        return {
            "total_checks": 0,
            "drift_count": 0,
            "drift_rate": 0.0,
            "mean_ks_statistic": 0.0,
            "min_p_value": 1.0,
        }
    total = len(drift_results)
    drifted = [r for r in drift_results if r.get("drift_detected")]
    ks_values = [float(r.get("ks_statistic", 0.0)) for r in drift_results]
    p_values = [float(r.get("p_value", 1.0)) for r in drift_results]
    return {
        "total_checks": total,
        "drift_count": len(drifted),
        "drift_rate": round(len(drifted) / total, 4),
        "mean_ks_statistic": round(sum(ks_values) / total, 4),
        "min_p_value": round(min(p_values), 4),
    }


def get_reference_window_size() -> int:
    """Return the current number of samples in the in-memory reference window."""
    return len(_reference_window)


def is_reference_window_ready(min_samples: int = 10) -> bool:
    """Return True if the reference window has at least *min_samples* readings.

    Args:
        min_samples: Minimum number of samples required (default 10).

    Returns:
        True if the window is large enough for drift detection.
    """
    return len(_reference_window) >= min_samples


def reference_window_stats() -> dict[str, Any]:
    """Return descriptive statistics for the in-memory reference window."""
    vals = _reference_window
    if not vals:
        return {"size": 0, "mean": None, "min": None, "max": None, "std": None}
    n = len(vals)
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    return {
        "size": n,
        "mean": round(mean, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "std": round(variance ** 0.5, 4),
    }
