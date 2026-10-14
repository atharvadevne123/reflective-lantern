"""Airflow champion/challenger retraining DAG for Trade-Sage."""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    HAS_AIRFLOW = True
except ImportError:
    HAS_AIRFLOW = False
    DAG = None
    PythonOperator = None

CHAMPION_PATH = os.getenv("MODEL_PATH", "model.joblib")
CHALLENGER_PATH = os.getenv("CHALLENGER_MODEL_PATH", "model_challenger.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "metrics.json")
CHALLENGER_METRICS_PATH = "metrics_challenger.json"
AUC_GATE = float(os.getenv("AUC_GATE", "0.60"))


def _load_training_data():
    """Return (X, y) — replace with real data source in production."""
    import numpy as np

    from app.features import FEATURE_NAMES

    rng = np.random.default_rng(int(datetime.utcnow().timestamp()) % 10000)
    n = 3000
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    score = 0.4 * X[:, 0] + 0.3 * X[:, 7] + 0.3 * rng.standard_normal(n)
    y = (score > 0).astype(int)
    return X, y


def task_read_champion_auc(**ctx) -> float:
    from app.model import read_champion_auc

    auc = read_champion_auc()
    logger.info("Champion AUC before retraining: %.4f", auc)
    return auc


def task_train_challenger(**ctx) -> dict:
    import os

    os.environ["MODEL_PATH"] = CHALLENGER_PATH
    os.environ["METRICS_PATH"] = CHALLENGER_METRICS_PATH

    from app.model import train_model

    X, y = _load_training_data()
    _, metrics = train_model(X, y)
    logger.info("Challenger AUC: %.4f ± %.4f", metrics["auc_mean"], metrics["auc_std"])
    return metrics


def task_compare_and_promote(**ctx):
    ti = ctx["ti"]
    champion_auc = ti.xcom_pull(task_ids="read_champion_auc")
    challenger_metrics = ti.xcom_pull(task_ids="train_challenger")

    challenger_auc = challenger_metrics["auc_mean"]
    logger.info("Champion AUC: %.4f, Challenger AUC: %.4f", champion_auc, challenger_auc)

    if challenger_auc >= max(champion_auc, AUC_GATE):
        shutil.copy(CHALLENGER_PATH, CHAMPION_PATH)
        shutil.copy(CHALLENGER_METRICS_PATH, METRICS_PATH)
        logger.info("Challenger PROMOTED (AUC %.4f > %.4f)", challenger_auc, champion_auc)
        return "promoted"
    else:
        logger.info("Challenger REJECTED (AUC %.4f < %.4f)", challenger_auc, champion_auc)
        for p in [CHALLENGER_PATH, CHALLENGER_METRICS_PATH]:
            Path(p).unlink(missing_ok=True)
        return "rejected"


def task_validate_gate(**ctx):
    ti = ctx["ti"]
    outcome = ti.xcom_pull(task_ids="compare_and_promote")
    if outcome == "rejected":
        logger.warning("Retraining gate not met — champion retained")
    else:
        logger.info("New champion deployed successfully")


if HAS_AIRFLOW and DAG is not None:
    default_args = {
        "owner": "trade-sage",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="trade_sage_retrain",
        default_args=default_args,
        description="Weekly champion/challenger retraining for Trade-Sage",
        schedule_interval="@weekly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["trade-sage", "ml", "retraining"],
    ) as dag:
        t1 = PythonOperator(task_id="read_champion_auc", python_callable=task_read_champion_auc)
        t2 = PythonOperator(task_id="train_challenger", python_callable=task_train_challenger)
        t3 = PythonOperator(task_id="compare_and_promote", python_callable=task_compare_and_promote)
        t4 = PythonOperator(task_id="validate_gate", python_callable=task_validate_gate)

        t1 >> t2 >> t3 >> t4
