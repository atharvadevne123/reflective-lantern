"""Prediction logging and KS-test drift detection."""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Any

from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, Prediction

logger = logging.getLogger(__name__)

# In-memory rolling buffer for reference window (populated at startup)
_REFERENCE_BUFFER: dict[str, deque] = {
    "distance_km": deque(maxlen=500),
    "weight_kg": deque(maxlen=500),
    "predicted_minutes": deque(maxlen=500),
}


def log_prediction(
    db: Session,
    *,
    carrier: str,
    distance_km: float,
    weight_kg: float,
    route_type: str,
    hour_of_day: int,
    day_of_week: int,
    predicted_minutes: float,
    confidence: float,
    model_version: str = "1.0.0",
) -> Prediction:
    """Persist a prediction record and update reference buffers."""
    pred = Prediction(
        created_at=datetime.utcnow(),
        carrier=carrier,
        distance_km=distance_km,
        weight_kg=weight_kg,
        route_type=route_type,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        predicted_minutes=predicted_minutes,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    _REFERENCE_BUFFER["distance_km"].append(distance_km)
    _REFERENCE_BUFFER["weight_kg"].append(weight_kg)
    _REFERENCE_BUFFER["predicted_minutes"].append(predicted_minutes)

    logger.debug("Logged prediction id=%d minutes=%.1f", pred.id, predicted_minutes)
    return pred


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Run KS test between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "reason": "insufficient data",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def run_drift_check(db: Session, current_window: int = 100) -> dict[str, Any]:
    """Compare latest predictions against reference buffer; log results."""
    recent = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(current_window)
        .all()
    )
    if not recent:
        return {"status": "no_predictions", "features": {}}

    results: dict[str, Any] = {}
    for feature in ("distance_km", "weight_kg", "predicted_minutes"):
        ref = list(_REFERENCE_BUFFER[feature])
        cur = [getattr(r, feature) for r in recent]
        drift = compute_drift(ref, cur)
        results[feature] = drift

        if drift.get("drift_detected"):
            log = DriftLog(
                feature=feature,
                ks_statistic=drift["ks_statistic"],
                p_value=drift["p_value"],
                drift_detected=1,
            )
            db.add(log)
            logger.warning("Drift detected in '%s': KS=%.4f p=%.4f", feature, drift["ks_statistic"], drift["p_value"])

    db.commit()
    return {"status": "ok", "features": results}


def seed_reference_buffer(samples: list[dict]) -> None:
    """Pre-load reference buffer from training data summary."""
    for s in samples:
        for k in _REFERENCE_BUFFER:
            if k in s:
                _REFERENCE_BUFFER[k].append(s[k])
    logger.info("Reference buffer seeded with %d samples", len(samples))
