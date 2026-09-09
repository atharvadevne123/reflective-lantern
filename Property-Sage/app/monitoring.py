"""Drift detection, prediction logging, and model monitoring."""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, Prediction

logger = logging.getLogger(__name__)

REFERENCE_PRICE_DISTRIBUTION = list(
    np.random.default_rng(0).normal(loc=450_000, scale=180_000, size=500).clip(50_000, 2_000_000)
)
REFERENCE_SQFT_DISTRIBUTION = list(
    np.random.default_rng(1).normal(loc=1500, scale=600, size=500).clip(300, 5000)
)


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """KS-test drift between a reference window and current predictions."""
    if len(current) < 10:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "sample_size": len(current)}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
        "sample_size": len(current),
    }


def log_prediction(
    db: Session,
    request_id: str,
    input_data: dict[str, Any],
    output: dict[str, float],
) -> None:
    record = Prediction(
        request_id=request_id,
        bedrooms=int(input_data["bedrooms"]),
        bathrooms=float(input_data["bathrooms"]),
        sqft=float(input_data["sqft"]),
        lot_size=float(input_data.get("lot_size", 5000.0)),
        year_built=int(input_data["year_built"]),
        neighborhood=str(input_data["neighborhood"]),
        property_type=str(input_data["property_type"]),
        predicted_price=output["predicted_price"],
        predicted_rental_yield=output["predicted_rental_yield"],
        confidence_score=None,
    )
    db.add(record)
    db.commit()
    logger.debug("Logged prediction %s", request_id)


def check_prediction_drift(db: Session) -> dict[str, Any]:
    since = datetime.utcnow() - timedelta(hours=24)
    recent = db.query(Prediction).filter(Prediction.created_at >= since).all()

    if not recent:
        return {"status": "no_recent_predictions", "checks": []}

    prices = [r.predicted_price for r in recent]
    sqfts = [r.sqft for r in recent]

    price_drift = compute_drift(REFERENCE_PRICE_DISTRIBUTION, prices)
    sqft_drift = compute_drift(REFERENCE_SQFT_DISTRIBUTION, sqfts)

    checks = [
        {"feature": "predicted_price", **price_drift},
        {"feature": "sqft_input", **sqft_drift},
    ]

    for check in checks:
        log = DriftLog(
            feature_name=check["feature"],
            ks_statistic=check["ks_statistic"],
            p_value=check["p_value"],
            drift_detected=int(check["drift_detected"]),
        )
        db.add(log)
    db.commit()

    any_drift = any(c["drift_detected"] for c in checks)
    logger.info("Drift check complete — drift_detected=%s", any_drift)
    return {"status": "ok", "drift_detected": any_drift, "checks": checks}


def get_prediction_stats(db: Session) -> dict[str, Any]:
    total = db.query(Prediction).count()
    since = datetime.utcnow() - timedelta(hours=24)
    last_24h = db.query(Prediction).filter(Prediction.created_at >= since).count()

    recent = db.query(Prediction).filter(Prediction.created_at >= since).all()
    avg_price = float(np.mean([r.predicted_price for r in recent])) if recent else 0.0
    avg_yield = float(np.mean([r.predicted_rental_yield for r in recent])) if recent else 0.0

    return {
        "total_predictions": total,
        "predictions_last_24h": last_24h,
        "avg_price_last_24h": round(avg_price, 2),
        "avg_rental_yield_last_24h": round(avg_yield, 4),
    }
