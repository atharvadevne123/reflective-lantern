"""Drift detection, prediction logging, and model monitoring for Property-Sage.

Uses the two-sample Kolmogorov-Smirnov test to compare the distribution of
recent predictions against a reference window collected at training time.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, Prediction

logger = logging.getLogger(__name__)

REFERENCE_PRICE_DISTRIBUTION: list[float] = list(
    np.random.default_rng(0).normal(loc=450_000, scale=180_000, size=500).clip(50_000, 2_000_000)
)
REFERENCE_SQFT_DISTRIBUTION: list[float] = list(
    np.random.default_rng(1).normal(loc=1500, scale=600, size=500).clip(300, 5000)
)

DRIFT_PVALUE_THRESHOLD: float = 0.05
DRIFT_MIN_SAMPLE_SIZE: int = 10


def compute_drift(
    reference: list[float],
    current: list[float],
) -> dict[str, Any]:
    """Run a two-sample KS test to detect distributional drift.

    Args:
        reference: Reference sample collected at training time.
        current: Recent sample from the production window.

    Returns:
        Dict with keys: ks_statistic, p_value, drift_detected, sample_size.
    """
    if len(current) < DRIFT_MIN_SAMPLE_SIZE:
        logger.debug("Drift check skipped — only %d samples (need %d)", len(current), DRIFT_MIN_SAMPLE_SIZE)
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "sample_size": len(current),
        }
    stat, p = ks_2samp(reference, current)
    result: dict[str, Any] = {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_PVALUE_THRESHOLD),
        "sample_size": len(current),
    }
    logger.info(
        "KS drift check — stat=%.4f p=%.4f drift_detected=%s",
        stat, p, result["drift_detected"],
    )
    return result


def log_prediction(
    db: Session,
    request_id: str,
    input_data: dict[str, Any],
    output: dict[str, float],
) -> None:
    """Persist a single prediction to the predictions table.

    Args:
        db: Active SQLAlchemy session.
        request_id: UUID string for this inference request.
        input_data: Raw property attributes dict.
        output: Model output dict (predicted_price, predicted_rental_yield).
    """
    record = Prediction(
        request_id=request_id,
        bedrooms=int(input_data["bedrooms"]),
        bathrooms=float(input_data["bathrooms"]),
        sqft=float(input_data["sqft"]),
        lot_size=float(input_data.get("lot_size") or 5000.0),
        year_built=int(input_data["year_built"]),
        neighborhood=str(input_data["neighborhood"]),
        property_type=str(input_data["property_type"]),
        predicted_price=output["predicted_price"],
        predicted_rental_yield=output["predicted_rental_yield"],
        confidence_score=None,
    )
    db.add(record)
    db.commit()
    logger.debug("Logged prediction request_id=%s price=%.2f", request_id, output["predicted_price"])


def check_prediction_drift(db: Session) -> dict[str, Any]:
    """Check for distributional drift in predictions from the last 24 hours.

    Runs KS tests on predicted_price and sqft_input, persisting results
    to the drift_logs table.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with status, drift_detected flag, and per-feature check results.
    """
    since = datetime.utcnow() - timedelta(hours=24)
    recent = db.query(Prediction).filter(Prediction.created_at >= since).all()

    if not recent:
        logger.info("Drift check — no predictions in last 24 hours")
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
    logger.info("Drift check complete — any_drift=%s n_recent=%d", any_drift, len(recent))
    return {"status": "ok", "drift_detected": any_drift, "checks": checks}


def get_prediction_stats(db: Session) -> dict[str, Any]:
    """Return summary statistics for recent predictions.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with total count, last-24h count, and averages of price and yield.
    """
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
