"""Drift detection, prediction logging, and model monitoring."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftLog, ModelMetrics, PredictionLog

logger = logging.getLogger(__name__)

DRIFT_P_THRESHOLD = float(os.getenv("DRIFT_P_THRESHOLD", "0.05"))
REFERENCE_WINDOW = int(os.getenv("REFERENCE_WINDOW", "500"))

_reference_store: dict[str, list[float]] = {}


def compute_drift(reference: list[float], current: list[float]) -> dict:
    """KS-test between reference and current feature distributions."""
    if len(reference) < 10 or len(current) < 2:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < DRIFT_P_THRESHOLD),
    }


def update_reference(feature_name: str, values: list[float]) -> None:
    """Append new observations to the rolling reference window for a feature."""
    buf = _reference_store.setdefault(feature_name, [])
    buf.extend(values)
    if len(buf) > REFERENCE_WINDOW:
        _reference_store[feature_name] = buf[-REFERENCE_WINDOW:]


def clear_reference_store() -> None:
    """Wipe the in-memory reference store (e.g. between test runs or model retrains)."""
    _reference_store.clear()


def get_drift_summary() -> dict[str, dict]:
    """Return the latest drift status for all tracked features.

    Returns:
        Dict mapping feature name to its most recent drift result.
        Empty when no drift has been computed yet.
    """
    summary: dict[str, dict] = {}
    for name, values in _reference_store.items():
        if len(values) >= 10:
            mid = len(values) // 2
            ref = values[:mid]
            cur = values[mid:]
            summary[name] = compute_drift(ref, cur)
    return summary


def check_drift_all(
    current_vectors: np.ndarray,
    feature_names: list[str],
    db: Session | None = None,
) -> dict[str, dict]:
    results = {}
    for i, name in enumerate(feature_names):
        col = current_vectors[:, i].tolist()
        ref = _reference_store.get(name, [])
        drift_info = compute_drift(ref, col)
        results[name] = drift_info
        update_reference(name, col)
        if db is not None:
            db.add(
                DriftLog(
                    feature_name=name,
                    ks_statistic=drift_info["ks_statistic"],
                    p_value=drift_info["p_value"],
                    drift_detected=drift_info["drift_detected"],
                )
            )
    if db is not None:
        try:
            db.commit()
        except Exception:
            db.rollback()
    return results


def log_prediction(
    db: Session,
    ticker: str,
    prediction: int,
    confidence: float,
    horizon_days: int,
    features: dict | None = None,
) -> None:
    record = PredictionLog(
        ticker=ticker,
        prediction=prediction,
        confidence=round(confidence, 4),
        horizon_days=horizon_days,
        features_json=json.dumps(features) if features else None,
    )
    db.add(record)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


def record_training_run(db: Session, metrics: dict) -> None:
    db.add(
        ModelMetrics(
            run_id=metrics.get("run_id", "unknown"),
            auc_mean=metrics.get("auc_mean", 0.0),
            auc_std=metrics.get("auc_std", 0.0),
            accuracy=metrics.get("accuracy"),
            n_features=metrics.get("n_features", 0),
            n_samples=metrics.get("n_samples", 0),
        )
    )
    try:
        db.commit()
    except Exception:
        db.rollback()
