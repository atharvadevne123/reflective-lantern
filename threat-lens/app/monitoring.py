"""Drift detection and prediction logging for Threat-Lens."""

import logging
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)

REFERENCE_WINDOW = 500
DRIFT_THRESHOLD = 0.05


def log_prediction(
    db: Session,
    correlation_id: str,
    flow: dict[str, Any],
    result: dict[str, Any],
) -> PredictionLog:
    """Persist a prediction result to the database."""
    record = PredictionLog(
        correlation_id=correlation_id,
        src_bytes=float(flow.get("src_bytes", 0)),
        dst_bytes=float(flow.get("dst_bytes", 0)),
        duration=float(flow.get("duration", 0)),
        protocol_type=str(flow.get("protocol_type", "")),
        service=str(flow.get("service", "")),
        flag=str(flow.get("flag", "")),
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        is_attack=result["is_attack"],
        raw_features=str(flow),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.debug("Logged prediction id=%d class=%s", record.id, record.predicted_class)
    return record


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run a KS-test comparing reference and current feature distributions.

    Args:
        reference: Historical feature values (baseline window).
        current: Recent feature values to compare against.

    Returns:
        Dict with ks_statistic, p_value, and drift_detected flag.
    """
    if len(reference) < 2 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}
    stat, p = ks_2samp(reference, current)
    drift = bool(p < DRIFT_THRESHOLD)
    logger.info("KS-test: stat=%.4f p=%.4f drift=%s", stat, p, drift)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": drift,
    }


def run_full_drift_check(
    db: Session,
    reference_data: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """Compute KS drift for each feature using recent prediction logs.

    Args:
        db: Active database session.
        reference_data: Mapping of feature_name → list of reference values.

    Returns:
        List of drift result dicts, one per feature.
    """
    recent = (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(REFERENCE_WINDOW)
        .all()
    )
    if not recent:
        logger.warning("No recent predictions found for drift check")
        return []

    results: list[dict[str, Any]] = []
    for feature, ref_values in reference_data.items():
        current_values = _extract_feature(recent, feature)
        if not current_values:
            continue
        drift = compute_drift(ref_values, current_values)
        report = DriftReport(
            feature_name=feature,
            ks_statistic=drift["ks_statistic"],
            p_value=drift["p_value"],
            drift_detected=int(drift["drift_detected"]),
            reference_n=len(ref_values),
            current_n=len(current_values),
        )
        db.add(report)
        results.append({"feature": feature, **drift})

    db.commit()
    drifted = [r for r in results if r["drift_detected"]]
    logger.info(
        "Drift check: %d/%d features drifted", len(drifted), len(results)
    )
    return results


def _extract_feature(logs: list[PredictionLog], feature: str) -> list[float]:
    """Extract a named numeric column from prediction log rows."""
    mapping = {
        "src_bytes": lambda r: r.src_bytes,
        "dst_bytes": lambda r: r.dst_bytes,
        "duration": lambda r: r.duration,
        "confidence": lambda r: r.confidence,
    }
    getter = mapping.get(feature)
    if getter is None:
        return []
    return [getter(r) for r in logs if getter(r) is not None]


def get_drift_summary(db: Session, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent drift reports."""
    rows = (
        db.query(DriftReport)
        .order_by(DriftReport.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "feature": r.feature_name,
            "ks_statistic": r.ks_statistic,
            "p_value": r.p_value,
            "drift_detected": bool(r.drift_detected),
        }
        for r in rows
    ]
