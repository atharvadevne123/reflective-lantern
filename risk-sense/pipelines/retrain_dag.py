"""Automated retraining pipeline for Risk-Sense model."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def fetch_training_data(db_session: Any, min_records: int = 500) -> tuple[np.ndarray, np.ndarray] | None:
    """Fetch labeled training data from database; return None if insufficient."""
    from app.database import CreditApplication

    records = db_session.query(CreditApplication).all()
    if len(records) < min_records:
        logger.warning("Insufficient training data: %d < %d required", len(records), min_records)
        return None

    X_rows = []
    y_rows = []
    for r in records:
        if r.default_probability is None:
            continue
        X_rows.append(
            [
                r.debt_to_income or 0.0,
                r.payment_history_score or 0.0,
                r.credit_utilization or 0.0,
                r.account_age_months or 1,
                r.num_open_accounts or 1,
                r.num_derogatory_marks or 0,
                r.annual_income or 50_000.0,
            ]
        )
        y_rows.append(1 if r.default_probability > 0.5 else 0)

    return np.array(X_rows), np.array(y_rows)


def should_retrain(db_session: Any, drift_threshold: float = 0.25) -> bool:
    """Return True if recent predictions show significant drift warranting retraining."""
    from app.database import CreditApplication
    from app.monitoring import ks_drift_test

    records = db_session.query(CreditApplication).order_by(CreditApplication.created_at.asc()).all()

    if len(records) < 40:
        return False

    mid = len(records) // 2
    reference = [r.default_probability for r in records[:mid] if r.default_probability is not None]
    current = [r.default_probability for r in records[mid:] if r.default_probability is not None]

    result = ks_drift_test(reference, current)
    drift_detected = result.get("drift_detected", False)
    ks_stat = result.get("ks_statistic", 0.0)

    if drift_detected and ks_stat > drift_threshold:
        logger.info("Drift detected (KS=%.4f) — retraining recommended", ks_stat)
        return True
    return False


def run_retrain_pipeline(db_session: Any) -> dict[str, Any]:
    """Main retraining pipeline: check drift, fetch data, train, save."""
    from app.features import build_pipeline
    from app.model import save_model, train

    result: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "retrained": False,
        "metrics": None,
        "reason": None,
    }

    if not should_retrain(db_session):
        result["reason"] = "no_drift_detected"
        return result

    data = fetch_training_data(db_session)
    if data is None:
        result["reason"] = "insufficient_data"
        return result

    X, y = data
    pipeline = build_pipeline()
    model, metrics = train(X, y, pipeline)
    save_model(model, pipeline)

    result["retrained"] = True
    result["metrics"] = metrics
    result["reason"] = "drift_triggered"
    logger.info("Retraining complete: %s", metrics)
    return result


def run_weekly_retrain(db_session: Any) -> dict[str, Any]:
    """Force retraining regardless of drift (weekly scheduled run)."""
    from app.features import build_pipeline
    from app.model import generate_synthetic_data, save_model, train

    data = fetch_training_data(db_session, min_records=100)
    if data is None:
        X, y = generate_synthetic_data(n_samples=2000)
        logger.info("Using synthetic data for weekly retrain")
    else:
        X, y = data

    pipeline = build_pipeline()
    model, metrics = train(X, y, pipeline)
    save_model(model, pipeline)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "retrained": True,
        "metrics": metrics,
        "reason": "scheduled_weekly",
    }
