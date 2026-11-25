"""Automated retraining pipeline (Airflow DAG) for Support-Sage."""
from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

DEFAULT_ARGS = {
    "owner": "support-sage",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def task_generate_data(**context: dict) -> dict:
    """Generate fresh synthetic training data."""
    import sys
    sys.path.insert(0, ".")
    from app.features import generate_synthetic_data
    df = generate_synthetic_data(n_samples=3000)
    df.to_parquet("/tmp/training_data.parquet", index=False)
    return {"n_samples": len(df)}


def task_train_model(**context: dict) -> dict:
    """Train the ensemble model on the generated data."""
    import sys

    import pandas as pd
    sys.path.insert(0, ".")
    from app.model import train_model
    df = pd.read_parquet("/tmp/training_data.parquet")
    _, metrics = train_model(df)
    return metrics


def task_rebuild_rag_index(**context: dict) -> dict:
    """Rebuild the FAISS retrieval index."""
    import sys

    import pandas as pd
    sys.path.insert(0, ".")
    from rag.ingest import ingest_tickets
    df = pd.read_parquet("/tmp/training_data.parquet")
    n = ingest_tickets(df)
    return {"indexed_tickets": n}


def task_check_drift(**context: dict) -> dict:
    """Run KS-test drift check and log results."""
    import sys
    sys.path.insert(0, ".")
    from app.database import SessionLocal
    from app.monitoring import get_drift_report
    db = SessionLocal()
    try:
        report = get_drift_report(db)
    finally:
        db.close()
    if report.get("any_drift_detected"):
        print("Drift detected - retraining scheduled")
    return report


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="support_sage_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly retraining and drift check for Support-Sage",
        schedule_interval="@weekly",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["ml", "support-sage"],
    ) as dag:
        generate = PythonOperator(task_id="generate_data", python_callable=task_generate_data)
        train = PythonOperator(task_id="train_model", python_callable=task_train_model)
        rebuild_rag = PythonOperator(task_id="rebuild_rag_index", python_callable=task_rebuild_rag_index)
        check = PythonOperator(task_id="check_drift", python_callable=task_check_drift)

        generate >> train >> rebuild_rag >> check
