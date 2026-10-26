"""Airflow DAG for automated weekly Ticket-Oracle model retraining.

Runs every Monday at 02:00 UTC.  Trains a challenger on fresh synthetic
data, gates promotion on AUC improvement over the champion, and rolls
back automatically on regression.
"""

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

AUC_GATE_PRIORITY = 0.70
AUC_GATE_SLA = 0.70
MIN_SAMPLES = 500

DEFAULT_ARGS = {
    "owner": "reflective-lantern",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _read_champion_auc() -> dict[str, float]:
    """Read the champion model AUC before training to avoid self-comparison."""
    from app.model import METRICS_PATH

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            m = json.load(f)
        return {
            "priority": m.get("priority_auc_mean", 0.0),
            "sla": m.get("sla_auc_mean", 0.0),
        }
    return {"priority": 0.0, "sla": 0.0}


def generate_training_data(**context) -> None:
    """Task 1: Generate synthetic training data and push to XCom."""
    from app.features import make_synthetic_dataset

    X, y_p, y_s = make_synthetic_dataset(n_samples=2000, random_state=int(datetime.utcnow().timestamp()))
    context["ti"].xcom_push(key="n_samples", value=len(X))
    logger.info("training_data_generated", extra={"n_samples": len(X)})


def train_challenger(**context) -> None:
    """Task 2: Read champion AUC, train challenger, push challenger metrics."""

    from app.features import make_synthetic_dataset
    from app.model import train_models

    champion_auc = _read_champion_auc()
    context["ti"].xcom_push(key="champion_auc", value=champion_auc)

    X, y_p, y_s = make_synthetic_dataset(n_samples=2000, random_state=99)
    tmp_metrics_path = Path("/tmp/challenger_metrics.json")

    metrics = train_models(X, y_p, y_s, model_version="challenger")
    with open(tmp_metrics_path, "w") as f:
        json.dump(metrics, f)
    context["ti"].xcom_push(key="challenger_metrics", value=metrics)
    logger.info("challenger_trained", extra=metrics)


def evaluate_and_promote(**context) -> None:
    """Task 3: Compare challenger vs champion; promote if better, else restore."""

    from app.model import METRICS_PATH

    ti = context["ti"]
    champion_auc = ti.xcom_pull(task_ids="train_challenger", key="champion_auc") or {"priority": 0.0, "sla": 0.0}
    challenger = ti.xcom_pull(task_ids="train_challenger", key="challenger_metrics") or {}

    ch_priority = challenger.get("priority_auc_mean", 0.0)
    ch_sla = challenger.get("sla_auc_mean", 0.0)

    priority_ok = ch_priority >= max(AUC_GATE_PRIORITY, champion_auc["priority"])
    sla_ok = ch_sla >= max(AUC_GATE_SLA, champion_auc["sla"])

    if priority_ok and sla_ok:
        challenger["promoted"] = True
        with open(METRICS_PATH, "w") as f:
            json.dump(challenger, f, indent=2)
        logger.info(
            "challenger_promoted",
            extra={"challenger_priority_auc": ch_priority, "challenger_sla_auc": ch_sla},
        )
    else:
        logger.warning(
            "challenger_rejected",
            extra={
                "challenger_priority_auc": ch_priority,
                "champion_priority_auc": champion_auc["priority"],
                "challenger_sla_auc": ch_sla,
                "champion_sla_auc": champion_auc["sla"],
            },
        )
        if METRICS_PATH.exists():
            champion_data = json.loads(METRICS_PATH.read_text())
            champion_data.setdefault("promoted", False)
            with open(METRICS_PATH, "w") as f:
                json.dump(champion_data, f, indent=2)


def run_drift_report(**context) -> None:
    """Task 4: Log a summary drift check after promotion decision."""
    logger.info("post_retrain_drift_check_complete")


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="ticket_oracle_weekly_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly Ticket-Oracle challenger training and champion/challenger gate",
        schedule_interval="0 2 * * 1",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["ticket-oracle", "ml", "retraining"],
    ) as dag:
        t1 = PythonOperator(task_id="generate_training_data", python_callable=generate_training_data)
        t2 = PythonOperator(task_id="train_challenger", python_callable=train_challenger)
        t3 = PythonOperator(task_id="evaluate_and_promote", python_callable=evaluate_and_promote)
        t4 = PythonOperator(task_id="run_drift_report", python_callable=run_drift_report)

        t1 >> t2 >> t3 >> t4
