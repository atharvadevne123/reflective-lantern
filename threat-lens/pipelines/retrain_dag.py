"""Airflow DAG for automated Threat-Lens model retraining."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

RETRAINING_THRESHOLD = 0.05  # drift p-value threshold
MIN_NEW_SAMPLES = 200  # minimum new records before retraining

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "threat-lens",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift(**context: Any) -> bool:
    """Check recent prediction logs for feature drift.

    Pushes drift_detected flag to XCom for downstream tasks.
    """
    from app.database import SessionLocal  # noqa: PLC0415
    from app.monitoring import run_full_drift_check  # noqa: PLC0415

    db = SessionLocal()
    try:
        reference: dict[str, list[float]] = _load_reference_window()
        reports = run_full_drift_check(db, reference)
        drifted = [r for r in reports if r.get("drift_detected")]
        drift_detected = len(drifted) > 0
        logger.info("Drift check: %d features drifted", len(drifted))
        if "ti" in context:
            context["ti"].xcom_push(key="drift_detected", value=drift_detected)
        return drift_detected
    finally:
        db.close()


def collect_new_samples(**context: Any) -> int:
    """Extract recent labelled samples from prediction logs.

    Returns the count of new samples available for retraining.
    """
    from app.database import PredictionLog, SessionLocal  # noqa: PLC0415

    db = SessionLocal()
    try:
        count = db.query(PredictionLog).count()
        logger.info("New samples available: %d", count)
        if "ti" in context:
            context["ti"].xcom_push(key="n_samples", value=count)
        return count
    finally:
        db.close()


def retrain_model(**context: Any) -> dict[str, Any]:
    """Re-train model if drift detected or sample threshold reached."""
    from app.database import RetrainingEvent, SessionLocal  # noqa: PLC0415
    from app.features import generate_synthetic_dataset  # noqa: PLC0415
    from app.model import load_metrics, train_model  # noqa: PLC0415

    old_metrics = load_metrics()
    auc_before = old_metrics.get("accuracy_mean", 0.0)

    X, y = generate_synthetic_dataset(n_samples=3000)
    pipe, new_metrics = train_model(X, y)
    auc_after = new_metrics.get("accuracy_mean", 0.0)

    ti = context.get("ti")
    n_samples = ti.xcom_pull(key="n_samples") if ti else 0

    db = SessionLocal()
    try:
        event = RetrainingEvent(
            trigger_reason="scheduled_drift_check",
            auc_before=auc_before,
            auc_after=auc_after,
            n_samples=int(n_samples or 0),
            success=1,
        )
        db.add(event)
        db.commit()
        logger.info("Retraining complete: %.4f → %.4f", auc_before, auc_after)
    finally:
        db.close()

    return {"auc_before": auc_before, "auc_after": auc_after}


def _load_reference_window() -> dict[str, list[float]]:
    """Load or build a reference feature window for KS-test."""
    from app.features import generate_synthetic_dataset  # noqa: PLC0415

    X, _ = generate_synthetic_dataset(n_samples=500, seed=0)
    return {
        "src_bytes": X[:, 1].tolist(),
        "dst_bytes": X[:, 2].tolist(),
        "duration": X[:, 0].tolist(),
        "confidence": [0.9] * len(X),
    }


def run_retraining_pipeline() -> None:
    """Standalone runner for environments without Airflow."""
    logger.info("Starting manual retraining pipeline")
    drift = check_drift()
    n = collect_new_samples()
    if drift or n >= MIN_NEW_SAMPLES:
        result = retrain_model()
        logger.info("Retraining done: %s", result)
    else:
        logger.info("No retraining needed (drift=%s, n=%d)", drift, n)


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="threat_lens_retrain",
        default_args=DEFAULT_ARGS,
        description="Automated retraining for Threat-Lens intrusion detection model",
        schedule_interval="0 2 * * *",  # daily at 02:00 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["ml", "threat-lens", "retraining"],
    ) as dag:
        t_drift = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift,
        )
        t_collect = PythonOperator(
            task_id="collect_new_samples",
            python_callable=collect_new_samples,
        )
        t_retrain = PythonOperator(
            task_id="retrain_model",
            python_callable=retrain_model,
        )
        t_drift >> t_collect >> t_retrain
