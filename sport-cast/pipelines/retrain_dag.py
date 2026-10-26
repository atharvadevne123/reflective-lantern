"""Airflow DAG for automated Sport-Cast model retraining."""
from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

MIN_AUC_GATE = 0.60
MIN_SAMPLES = 500


def _fetch_training_data(**context) -> int:
    from app.features import make_synthetic_dataset

    X, y = make_synthetic_dataset(n=2000, seed=int(datetime.now().strftime("%Y%W")))
    import joblib

    joblib.dump((X, y), "/tmp/sport_cast_retrain_data.joblib")
    logger.info("Training data fetched: %d samples", len(y))
    return len(y)


def _retrain_model(**context) -> dict:
    import joblib

    from app.model import train_model

    n_samples = context["ti"].xcom_pull(task_ids="fetch_data")
    if n_samples < MIN_SAMPLES:
        raise ValueError(f"Insufficient data: {n_samples} < {MIN_SAMPLES}")

    X, y = joblib.load("/tmp/sport_cast_retrain_data.joblib")
    _, _, metrics = train_model(X, y)
    logger.info("Retrain metrics: %s", metrics)
    return metrics


def _validate_model(**context) -> None:
    import json
    from pathlib import Path

    metrics_path = Path("metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError("metrics.json not found after retraining")
    metrics = json.loads(metrics_path.read_text())
    auc = metrics.get("auc_mean", 0.0)
    if auc < MIN_AUC_GATE:
        raise ValueError(f"Model AUC {auc:.4f} below gate {MIN_AUC_GATE}")
    logger.info("Validation passed: AUC=%.4f", auc)


if DAG is not None:
    default_args = {
        "owner": "sport-cast",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="sport_cast_retrain",
        default_args=default_args,
        description="Weekly Sport-Cast model retraining with champion/challenger gate",
        schedule_interval="0 3 * * 1",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["sport-cast", "ml", "retraining"],
    ) as dag:
        fetch_data = PythonOperator(task_id="fetch_data", python_callable=_fetch_training_data)
        retrain = PythonOperator(task_id="retrain_model", python_callable=_retrain_model)
        validate = PythonOperator(task_id="validate_model", python_callable=_validate_model)
        fetch_data >> retrain >> validate
