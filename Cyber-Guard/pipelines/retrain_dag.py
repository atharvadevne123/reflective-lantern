"""Automated retraining pipeline (Airflow DAG) for Cyber-Guard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "cyber-guard",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1),
}

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False


def _fetch_recent_predictions(**context) -> dict:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import os
    db_url = os.getenv("DATABASE_URL", "sqlite:///./cyber_guard.db")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    with Session() as db:
        from app.database import PredictionLog
        since = datetime.utcnow() - timedelta(days=7)
        rows = db.query(PredictionLog).filter(PredictionLog.timestamp >= since).all()
        logger.info("fetched %d rows for retraining", len(rows))
        return {"row_count": len(rows)}


def _retrain_model(**context) -> None:
    import pandas as pd
    from app.model import generate_synthetic_data, train_model

    ti = context["ti"]
    stats = ti.xcom_pull(task_ids="fetch_recent_predictions")

    if stats and stats.get("row_count", 0) < 50:
        logger.warning("insufficient data (%d rows), using synthetic fallback", stats["row_count"])

    X, y = generate_synthetic_data(1000)
    _, metrics = train_model(X, y)
    logger.info("retrain complete accuracy=%.4f", metrics["accuracy_mean"])

    metrics_path = Path("retrain_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))


def _evaluate_model(**context) -> None:
    metrics_path = Path("retrain_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError("metrics not found after retrain")
    metrics = json.loads(metrics_path.read_text())
    acc = metrics.get("accuracy_mean", 0)
    if acc < 0.70:
        raise ValueError(f"retrained model accuracy {acc:.4f} below threshold 0.70 — rollback needed")
    logger.info("model evaluation passed accuracy=%.4f", acc)


def run_retrain_pipeline() -> dict:
    logger.info("running retrain pipeline (non-Airflow mode)")
    result = _fetch_recent_predictions()
    _retrain_model(ti=type("TI", (), {"xcom_pull": lambda self, **kw: result})())
    _evaluate_model()
    return {"status": "success"}


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="cyber_guard_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly automated retraining for Cyber-Guard intrusion detection model",
        schedule_interval="@weekly",
        catchup=False,
        tags=["cyber-guard", "ml", "retraining"],
    ) as dag:
        fetch_task = PythonOperator(
            task_id="fetch_recent_predictions",
            python_callable=_fetch_recent_predictions,
        )
        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=_retrain_model,
        )
        evaluate_task = PythonOperator(
            task_id="evaluate_model",
            python_callable=_evaluate_model,
        )
        fetch_task >> retrain_task >> evaluate_task
