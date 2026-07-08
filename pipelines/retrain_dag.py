"""Automated retraining pipeline — Airflow-compatible DAG pattern."""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DAG_ID = "traffic_pulse_retrain"
SCHEDULE_INTERVAL = "@weekly"
DEFAULT_ARGS: dict[str, Any] = {
    "owner": "reflective-lantern",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def task_check_drift(
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./traffic_pulse.db"),
) -> dict[str, Any]:
    """Query drift_logs for unresolved drift events."""
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.connect() as conn:
        try:
            row = conn.execute(text("SELECT COUNT(*) FROM drift_logs WHERE drift_detected = 1"))
            drift_count = row.scalar() or 0
        except Exception:
            drift_count = 0

    needs_retrain = drift_count > 0
    logger.info("Drift check drift_count=%d needs_retrain=%s", drift_count, needs_retrain)
    return {"drift_count": int(drift_count), "needs_retrain": needs_retrain}


def task_fetch_recent_data(
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./traffic_pulse.db"),
    days: int = 7,
) -> dict[str, int]:
    """Count recent predictions to size the retraining dataset."""
    from sqlalchemy import create_engine, text

    cutoff = datetime.utcnow() - timedelta(days=days)
    engine = create_engine(db_url)
    with engine.connect() as conn:
        try:
            row = conn.execute(
                text("SELECT COUNT(*) FROM prediction_logs WHERE timestamp > :cutoff"),
                {"cutoff": cutoff},
            )
            count = row.scalar() or 0
        except Exception:
            count = 0

    logger.info("Recent data count=%d over last %d days", count, days)
    return {"recent_prediction_count": int(count)}


def task_retrain(n_samples: int = 5000) -> dict[str, Any]:
    """Re-train the ensemble on fresh synthetic data."""
    from app.model import train_model

    logger.info("Starting retraining n_samples=%d", n_samples)
    _, metrics = train_model()
    logger.info("Retraining complete metrics=%s", metrics)
    return metrics


def task_validate_model(auc_threshold: float = 0.75) -> dict[str, Any]:
    """Assert retrained model meets minimum AUC before promotion."""
    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        return {"passed": False, "reason": "metrics.json not found"}

    metrics = json.loads(metrics_path.read_text())
    ensemble_auc = float(metrics.get("ensemble_auc", 0.0))
    passed = ensemble_auc >= auc_threshold
    return {"passed": passed, "ensemble_auc": ensemble_auc, "threshold": auc_threshold}


def task_promote_model(validation: dict[str, Any]) -> dict[str, str]:
    """Copy the validated model to the stable production path."""
    if not validation.get("passed"):
        logger.warning("Validation failed — skipping promotion: %s", validation)
        return {"status": "skipped", "reason": "validation_failed"}

    src = Path("model.joblib")
    dst = Path("model_stable.joblib")
    if src.exists():
        shutil.copy2(src, dst)
        logger.info("Promoted %s -> %s", src, dst)
        return {"status": "promoted", "model_path": str(dst)}
    return {"status": "skipped", "reason": "model_not_found"}


def run_pipeline(force: bool = False) -> dict[str, Any]:
    """Execute the full check-retrain-validate-promote pipeline."""
    results: dict[str, Any] = {}

    drift = task_check_drift()
    results["drift_check"] = drift

    if not force and not drift["needs_retrain"]:
        logger.info("No drift — skipping retraining")
        return {"status": "skipped", **results}

    data_info = task_fetch_recent_data()
    results["data_info"] = data_info

    n_samples = max(1000, min(data_info["recent_prediction_count"] * 10, 10_000))
    training_metrics = task_retrain(n_samples=n_samples)
    results["training_metrics"] = training_metrics

    validation = task_validate_model()
    results["validation"] = validation

    promotion = task_promote_model(validation)
    results["promotion"] = promotion

    return {"status": "completed", **results}


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    result = run_pipeline(force=force)
    print(json.dumps(result, indent=2, default=str))
