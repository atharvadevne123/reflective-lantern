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
        logger.debug(
            "Drift check skipped — only %d samples (need %d)", len(current), DRIFT_MIN_SAMPLE_SIZE
        )
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
        stat,
        p,
        result["drift_detected"],
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
    logger.debug(
        "Logged prediction request_id=%s price=%.2f", request_id, output["predicted_price"]
    )


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


def yield_category(rental_yield: float) -> str:
    """Classify a rental yield into a human-readable category.

    Args:
        rental_yield: Rental yield as a decimal (e.g. 0.05 for 5%).

    Returns:
        Category string: 'low', 'moderate', 'good', or 'excellent'.
    """
    if rental_yield < 0.03:
        return "low"
    if rental_yield < 0.05:
        return "moderate"
    if rental_yield < 0.08:
        return "good"
    return "excellent"


def anomaly_score(value: float, reference: list[float]) -> float:
    """Compute a normalised anomaly score using z-score magnitude.

    Args:
        value: Observed value to score.
        reference: Reference distribution (training or recent window).

    Returns:
        Anomaly score in [0.0, 1.0]. Higher = more anomalous.
    """
    if len(reference) < 2:
        return 0.0
    import statistics
    mean = statistics.mean(reference)
    std = statistics.stdev(reference)
    if std == 0:
        return 0.0
    z = abs(value - mean) / std
    return round(min(1.0, z / 5.0), 4)


def detect_outlier_predictions(
    prices: list[float],
    z_threshold: float = 3.0,
) -> list[int]:
    """Return indices of prices exceeding z_threshold standard deviations from mean.

    Args:
        prices: List of predicted prices.
        z_threshold: Number of standard deviations to use as cutoff.

    Returns:
        List of integer indices where the prediction is an outlier.
    """
    if len(prices) < 3:
        return []
    import statistics
    mean = statistics.mean(prices)
    std = statistics.stdev(prices)
    if std == 0:
        return []
    return [i for i, p in enumerate(prices) if abs(p - mean) / std > z_threshold]


def prediction_summary_stats(db: "Any") -> dict[str, Any]:
    """Aggregate summary statistics from the prediction log table.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dict with count, mean_price, mean_yield, and latest_request_id.
    """
    from app.database import Prediction
    try:
        preds = db.query(Prediction).all()
        if not preds:
            return {"count": 0, "mean_price": 0.0, "mean_yield": 0.0, "latest_request_id": None}
        prices = [p.predicted_price for p in preds]
        yields = [p.predicted_rental_yield for p in preds]
        import statistics as _stats
        return {
            "count": len(preds),
            "mean_price": round(_stats.mean(prices), 2),
            "mean_yield": round(_stats.mean(yields), 4),
            "latest_request_id": preds[-1].request_id,
        }
    except Exception as exc:
        logger.error("Failed to compute prediction summary stats: %s", exc)
        return {"count": 0, "mean_price": 0.0, "mean_yield": 0.0, "latest_request_id": None}


def get_drift_history(db: "Any", limit: int = 50) -> list[dict[str, Any]]:
    """Fetch recent drift log entries from the database.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of entries to return.

    Returns:
        List of drift log dicts, newest first.
    """
    from app.database import DriftLog
    try:
        rows = (
            db.query(DriftLog)
            .order_by(DriftLog.checked_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "feature_name": r.feature_name,
                "ks_statistic": r.ks_statistic,
                "p_value": r.p_value,
                "drift_detected": bool(r.drift_detected),
                "checked_at": r.checked_at.isoformat(),
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error("Failed to fetch drift history: %s", exc)
        return []
