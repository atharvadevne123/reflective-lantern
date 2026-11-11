"""Automated retraining pipeline (Airflow-compatible DAG)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False


def _retrain(**context) -> None:
    from app.model import prepare_training_data, save_model, train_model
    logger.info("Retraining started at %s", datetime.utcnow().isoformat())
    X, y = prepare_training_data(n_tickers=5)
    pipe, metrics = train_model(X, y)
    save_model(pipe, metrics)
    logger.info("Retraining complete: %s", metrics)
    Path("retrain_history.json").write_text(
        json.dumps({"timestamp": datetime.utcnow().isoformat(), **metrics})
    )


def _validate(**context) -> None:
    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found after retraining")
    metrics = json.loads(metrics_path.read_text())
    auc = metrics.get("auc_mean", 0)
    if auc < 0.52:
        raise ValueError(f"Model AUC {auc:.4f} below minimum threshold 0.52")
    logger.info("Model validation passed: AUC=%.4f", auc)


def run_pipeline() -> None:
    """Run the retraining pipeline directly (without Airflow)."""
    logger.info("Running retraining pipeline standalone")
    _retrain()
    _validate()
    logger.info("Pipeline complete")


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="alpha_scout_retrain",
        description="Weekly retraining of Alpha-Scout ML model",
        schedule_interval="@weekly",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args={
            "owner": "alpha-scout",
            "retries": 1,
            "retry_delay": timedelta(minutes=5),
        },
        tags=["ml", "retraining"],
    ) as dag:
        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=_retrain,
        )
        validate_task = PythonOperator(
            task_id="validate_model",
            python_callable=_validate,
        )
        retrain_task >> validate_task
