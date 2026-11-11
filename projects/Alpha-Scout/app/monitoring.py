"""Drift detection and prediction logging."""
from __future__ import annotations

import structlog
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = structlog.get_logger(__name__)

# Reference distribution seeded at startup from first N predictions
_reference_store: dict[str, list[float]] = {}


def log_prediction(
    db: Session,
    ticker: str,
    alpha_score: float,
    return_prediction: float,
    confidence: float,
    model_version: str = "1.0.0",
) -> PredictionLog:
    record = PredictionLog(
        ticker=ticker,
        alpha_score=alpha_score,
        return_prediction=return_prediction,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info("logged prediction ticker=%s alpha=%.2f", ticker, alpha_score)
    _update_reference(alpha_score)
    return record


def _update_reference(value: float, key: str = "alpha_score") -> None:
    _reference_store.setdefault(key, [])
    _reference_store[key].append(value)
    if len(_reference_store[key]) > 1000:
        _reference_store[key] = _reference_store[key][-1000:]


def compute_drift(reference: list[float], current: list[float]) -> dict:
    """KS-test drift detection between reference and current distributions."""
    if len(reference) < 10 or len(current) < 10:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "error": "insufficient samples",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def run_drift_check(db: Session, feature_name: str, current_values: list[float]) -> dict[str, object]:
    reference = _reference_store.get(feature_name, [])
    result = compute_drift(reference, current_values)
    record = DriftLog(
        feature_name=feature_name,
        ks_statistic=result["ks_statistic"],
        p_value=result["p_value"],
        drift_detected=int(result["drift_detected"]),
        sample_size=len(current_values),
    )
    db.add(record)
    db.commit()
    logger.info(
        "drift check feature=%s ks=%.4f p=%.4f detected=%s",
        feature_name,
        result["ks_statistic"],
        result["p_value"],
        result["drift_detected"],
    )
    return result


def get_prediction_metrics(db: Session) -> dict:
    total = db.query(PredictionLog).count()
    recent = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(100).all()
    if not recent:
        return {"total_predictions": total, "avg_alpha_score": None, "avg_confidence": None}
    avg_alpha = sum(r.alpha_score for r in recent) / len(recent)
    avg_conf = sum(r.confidence for r in recent) / len(recent)
    return {
        "total_predictions": total,
        "avg_alpha_score": round(avg_alpha, 2),
        "avg_confidence": round(avg_conf, 4),
        "recent_sample_size": len(recent),
    }


def check_pipeline_health(pipe) -> dict[str, object]:
    """Check whether a fitted sklearn pipeline passes basic sanity checks.

    Args:
        pipe: Fitted sklearn Pipeline.

    Returns:
        Dict with 'healthy' (bool) and diagnostics.
    """
    issues = []
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(pipe)
    except Exception as exc:
        issues.append(f"pipeline not fitted: {exc}")
    if not hasattr(pipe, "predict_proba"):
        issues.append("missing predict_proba")
    return {"healthy": len(issues) == 0, "issues": issues}
