"""Airflow DAG for automated Sports-Oracle model retraining."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

MIN_AUC = 0.65
MIN_SAMPLES = 500
CHAMPION_METRICS_PATH = Path("metrics.json")


def _load_champion_auc() -> float:
    """Read the champion model AUC before any training begins."""
    if CHAMPION_METRICS_PATH.exists():
        try:
            m = json.loads(CHAMPION_METRICS_PATH.read_text())
            return float(m.get("auc_mean", 0.0))
        except (json.JSONDecodeError, ValueError):
            pass
    return 0.0


def collect_training_data(**context) -> dict:
    """Collect and validate training data; push stats to XCom."""
    from app.features import make_synthetic_dataset

    X, y = make_synthetic_dataset(n=3000)
    n = len(y)
    if n < MIN_SAMPLES:
        raise ValueError(f"Insufficient training data: {n} < {MIN_SAMPLES}")

    logger.info("training_data_collected", extra={"n_samples": n})
    context["ti"].xcom_push(key="n_samples", value=n)
    return {"n_samples": n}


def train_challenger(**context) -> None:
    """Train the challenger model and push metrics to XCom."""
    from app.features import make_synthetic_dataset
    from app.model import train_model

    champion_auc = _load_champion_auc()
    context["ti"].xcom_push(key="champion_auc", value=champion_auc)

    X, y = make_synthetic_dataset(n=3000)
    _, metrics = train_model(X, y, model_path=Path("challenger.joblib"))

    context["ti"].xcom_push(key="challenger_auc", value=metrics["auc_mean"])
    context["ti"].xcom_push(key="challenger_metrics", value=metrics)
    logger.info("challenger_trained", extra={"auc": metrics["auc_mean"]})


def promote_champion(**context) -> None:
    """Promote challenger to champion if it meets the quality gate."""
    import shutil

    ti = context["ti"]
    challenger_auc: float = ti.xcom_pull(key="challenger_auc", task_ids="train_challenger")
    champion_auc: float = ti.xcom_pull(key="champion_auc", task_ids="train_challenger")

    logger.info(
        "promotion_gate",
        extra={"challenger_auc": challenger_auc, "champion_auc": champion_auc, "min_auc": MIN_AUC},
    )

    if challenger_auc >= MIN_AUC and challenger_auc >= champion_auc:
        challenger_path = Path("challenger.joblib")
        champion_path = Path("model.joblib")
        shutil.copy(challenger_path, champion_path)
        logger.info("challenger_promoted", extra={"auc": challenger_auc})
    else:
        logger.warning(
            "challenger_rejected",
            extra={"challenger_auc": challenger_auc, "required": max(MIN_AUC, champion_auc)},
        )
        # Restore champion metrics so they are not overwritten
        if CHAMPION_METRICS_PATH.exists():
            pass  # metrics.json was not overwritten; challenger.joblib was used


def drift_check(**context) -> None:
    """Run feature drift check and log results."""
    import numpy as np

    from app.monitoring import compute_drift

    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 500).tolist()
    current = rng.normal(0, 1, 200).tolist()
    result = compute_drift(reference, current)
    logger.info("scheduled_drift_check", extra=result)


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "sports-oracle",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="sports_oracle_retrain",
        default_args=default_args,
        description="Weekly model retraining for Sports-Oracle",
        schedule="0 2 * * 1",  # Every Monday at 02:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["ml", "sports-oracle"],
    ) as dag:
        t1 = PythonOperator(task_id="collect_data", python_callable=collect_training_data)
        t2 = PythonOperator(task_id="train_challenger", python_callable=train_challenger)
        t3 = PythonOperator(task_id="drift_check", python_callable=drift_check)
        t4 = PythonOperator(task_id="promote_champion", python_callable=promote_champion)

        t1 >> t2 >> [t3, t4]
