"""Model monitoring: KS-test drift detection, anomaly scoring, prediction logging."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Reference distribution (populated on first training)
_reference_distributions: dict[str, list[float]] = {}
_isolation_forest: IsolationForest | None = None


def set_reference_distributions(data: dict[str, list[float]]) -> None:
    global _reference_distributions
    _reference_distributions = {k: list(v) for k, v in data.items()}
    logger.info("Reference distributions set for %d features", len(data))


def compute_drift(reference: list[float], current: list[float]) -> dict:
    if len(reference) < 5 or len(current) < 5:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False, "error": "insufficient_data"}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
    }


def check_all_features(current_window: dict[str, list[float]]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for feature, current in current_window.items():
        ref = _reference_distributions.get(feature, [])
        results[feature] = compute_drift(ref, current)
    return results


def train_anomaly_detector(consumption_values: list[float]) -> IsolationForest:
    global _isolation_forest
    X = np.array(consumption_values).reshape(-1, 1)
    _isolation_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    _isolation_forest.fit(X)
    logger.info("IsolationForest trained on %d samples", len(consumption_values))
    return _isolation_forest


def detect_anomaly(consumption_kwh: float) -> dict:
    z_score = 0.0
    iso_score = 0.0
    is_iso_anomaly = False

    if _isolation_forest is not None:
        X = np.array([[consumption_kwh]])
        iso_score = float(-_isolation_forest.score_samples(X)[0])
        is_iso_anomaly = bool(_isolation_forest.predict(X)[0] == -1)

    ref_vals = _reference_distributions.get("consumption_kwh", [])
    if len(ref_vals) >= 10:
        mean = float(np.mean(ref_vals))
        std = max(float(np.std(ref_vals)), 1e-6)
        z_score = abs((consumption_kwh - mean) / std)

    is_anomaly = bool(is_iso_anomaly or z_score > 3.0)
    severity = "none"
    if z_score > 5.0 or iso_score > 0.6:
        severity = "critical"
    elif z_score > 3.0 or iso_score > 0.4:
        severity = "high"
    elif z_score > 2.0:
        severity = "medium"
    elif is_anomaly:
        severity = "low"

    anomaly_type = "none"
    ref_vals = _reference_distributions.get("consumption_kwh", [])
    if is_anomaly and ref_vals:
        mean = float(np.mean(ref_vals))
        anomaly_type = "spike" if consumption_kwh > mean else "drop"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(iso_score, 4),
        "z_score": round(z_score, 4),
        "severity": severity,
        "anomaly_type": anomaly_type,
    }


def log_prediction(
    db: Session,
    meter_id: str,
    predicted_kwh: float,
    features_used: dict,
    model_version: str = "1.0.0",
    confidence_score: float = 0.9,
    prediction_horizon_h: int = 1,
    actual_kwh: float | None = None,
) -> None:
    from app.database import PredictionLog

    record = PredictionLog(
        meter_id=meter_id,
        predicted_kwh=predicted_kwh,
        actual_kwh=actual_kwh,
        model_version=model_version,
        features_used=features_used,
        confidence_score=confidence_score,
        prediction_horizon_h=prediction_horizon_h,
        timestamp=datetime.utcnow(),
    )
    db.add(record)
    db.commit()


def log_anomaly(
    db: Session,
    meter_id: str,
    consumption_kwh: float,
    anomaly_result: dict,
) -> None:
    from app.database import AnomalyLog

    record = AnomalyLog(
        meter_id=meter_id,
        consumption_kwh=consumption_kwh,
        anomaly_score=anomaly_result["anomaly_score"],
        is_anomaly=anomaly_result["is_anomaly"],
        anomaly_type=anomaly_result["anomaly_type"],
        severity=anomaly_result["severity"],
        details=str(anomaly_result),
        timestamp=datetime.utcnow(),
    )
    db.add(record)
    db.commit()


def log_drift(db: Session, feature_results: dict[str, dict]) -> None:
    from app.database import DriftLog

    for feature, result in feature_results.items():
        if "error" in result:
            continue
        ref_vals = _reference_distributions.get(feature, [])
        record = DriftLog(
            feature_name=feature,
            ks_statistic=result["ks_statistic"],
            p_value=result["p_value"],
            drift_detected=result["drift_detected"],
            reference_mean=float(np.mean(ref_vals)) if ref_vals else 0.0,
            current_mean=0.0,
            timestamp=datetime.utcnow(),
        )
        db.add(record)
    db.commit()


def compute_prediction_stats(db: Session) -> dict:
    from app.database import PredictionLog

    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(200).all()
    if not logs:
        return {"count": 0}
    preds = [r.predicted_kwh for r in logs]
    return {
        "count": len(preds),
        "mean_kwh": round(float(np.mean(preds)), 4),
        "std_kwh": round(float(np.std(preds)), 4),
        "min_kwh": round(float(np.min(preds)), 4),
        "max_kwh": round(float(np.max(preds)), 4),
    }
