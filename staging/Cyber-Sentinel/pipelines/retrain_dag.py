"""Airflow DAG for automated Cyber-Sentinel model retraining."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    logger.warning("Airflow not installed; DAG definition skipped")


DEFAULT_ARGS = {
    "owner": "cyber-sentinel",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_drift(**context: object) -> dict[str, object]:
    """Check for drift in the latest prediction window.

    Returns a dict with drift flags for downstream tasks.
    """
    from app.monitoring import get_feature_drift_summary

    drift_summary = get_feature_drift_summary()
    drift_detected = any(
        v.get("drift_detected", False)
        for v in drift_summary.values()
        if isinstance(v, dict)
    )
    logger.info("Drift check: %s", "DRIFT" if drift_detected else "OK")
    return {"drift_detected": drift_detected, "summary": drift_summary}


def retrain_model(**context: object) -> dict[str, object]:
    """Retrain the ensemble model with fresh synthetic data.

    In production this would pull labeled data from the database.
    """
    from app.model import train_model

    logger.info("Starting scheduled retraining")
    metrics = train_model()
    logger.info("Retraining complete: F1=%.4f", metrics.get("f1_score", 0))
    return metrics


def update_reference_distributions(**context: object) -> None:
    """Regenerate reference distributions from the latest training data."""

    from app.features import generate_synthetic_data
    from app.monitoring import set_reference_distribution

    X, _ = generate_synthetic_data(n_samples=2000)
    n_features = X.shape[1]
    for i in range(n_features):
        set_reference_distribution(f"feature_{i}", X[:, i])
    logger.info("Reference distributions updated for %d features", n_features)


def send_alert(**context: object) -> None:
    """Log an alert if drift was detected (placeholder for notifications)."""
    ti = context.get("ti")
    drift_info = ti.xcom_pull(task_ids="check_drift") if ti else {}
    if drift_info and drift_info.get("drift_detected"):
        logger.warning("ALERT: Drift detected. Retraining was triggered.")


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="cyber_sentinel_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly retraining pipeline for Cyber-Sentinel",
        schedule_interval="@weekly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["ml", "retraining", "cyber-sentinel"],
    ) as dag:
        t_check_drift = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift,
        )

        t_retrain = PythonOperator(
            task_id="retrain_model",
            python_callable=retrain_model,
        )

        t_update_refs = PythonOperator(
            task_id="update_reference_distributions",
            python_callable=update_reference_distributions,
        )

        t_alert = PythonOperator(
            task_id="send_alert",
            python_callable=send_alert,
        )

        t_check_drift >> t_retrain >> t_update_refs >> t_alert
