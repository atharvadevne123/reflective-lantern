"""Automated retraining pipeline (Airflow DAG or standalone script)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
RETRAIN_THRESHOLD_AUC = float(os.getenv("RETRAIN_THRESHOLD_AUC", "0.75"))


def _load_current_metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def task_check_drift(**context) -> dict:
    """Read latest drift logs and decide whether retraining is needed.

    Returns:
        Dict with should_retrain flag and reason.
    """
    metrics = _load_current_metrics()
    breach_auc = metrics.get("breach_auc_mean", 1.0)
    should_retrain = breach_auc < RETRAIN_THRESHOLD_AUC

    result = {
        "should_retrain": should_retrain,
        "breach_auc_mean": breach_auc,
        "reason": f"AUC {breach_auc:.3f} below threshold {RETRAIN_THRESHOLD_AUC}" if should_retrain else "metrics healthy",
        "checked_at": datetime.utcnow().isoformat(),
    }
    logger.info("Drift/quality check: %s", result)
    return result


def task_generate_training_data(**context) -> int:
    """Generate fresh synthetic training data.

    Returns:
        Number of training samples generated.
    """
    from app.model import generate_synthetic_data

    df = generate_synthetic_data(n_samples=3000)
    data_path = MODEL_DIR / "train_data.parquet"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_path, index=False)
    logger.info("Training data written to %s (%d rows)", data_path, len(df))
    return len(df)


def task_retrain_models(**context) -> dict:
    """Retrain all three models and persist to MODEL_DIR.

    Returns:
        Cross-validation metrics.
    """
    import pandas as pd

    from app.model import train_models

    data_path = MODEL_DIR / "train_data.parquet"
    df = pd.read_parquet(data_path) if data_path.exists() else None
    metrics = train_models(df)
    logger.info("Retrain metrics: %s", metrics)
    return metrics


def task_validate_models(**context) -> bool:
    """Validate that retrained models meet minimum quality thresholds.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If models do not meet thresholds.
    """
    metrics = _load_current_metrics()
    auc = metrics.get("breach_auc_mean", 0.0)
    acc = metrics.get("category_accuracy_mean", 0.0)

    if auc < 0.60:
        raise ValueError(f"SLA breach AUC too low after retrain: {auc:.3f}")
    if acc < 0.50:
        raise ValueError(f"Category accuracy too low after retrain: {acc:.3f}")

    logger.info("Model validation passed: AUC=%.3f ACC=%.3f", auc, acc)
    return True


def run_pipeline() -> dict:
    """Execute the full retraining pipeline without Airflow.

    Returns:
        Final metrics dict.
    """
    logger.info("Starting standalone retrain pipeline")
    drift_result = task_check_drift()
    if not drift_result["should_retrain"]:
        logger.info("Retraining not required: %s", drift_result["reason"])
        return drift_result

    n_samples = task_generate_training_data()
    logger.info("Generated %d training samples", n_samples)
    metrics = task_retrain_models()
    task_validate_models()
    logger.info("Pipeline complete: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Airflow DAG definition (only when Airflow is installed)
# ---------------------------------------------------------------------------

if _AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "ticket-scout",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    dag = DAG(
        "ticket_scout_retrain",
        default_args=default_args,
        description="Nightly retraining pipeline for Ticket-Scout models",
        schedule_interval="0 2 * * *",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["ml", "ticket-scout"],
    )

    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=task_check_drift,
        dag=dag,
    )

    generate_data_task = PythonOperator(
        task_id="generate_training_data",
        python_callable=task_generate_training_data,
        dag=dag,
    )

    retrain_task = PythonOperator(
        task_id="retrain_models",
        python_callable=task_retrain_models,
        dag=dag,
    )

    validate_task = PythonOperator(
        task_id="validate_models",
        python_callable=task_validate_models,
        dag=dag,
    )

    check_drift_task >> generate_data_task >> retrain_task >> validate_task


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline()
    print(json.dumps(result, indent=2))
