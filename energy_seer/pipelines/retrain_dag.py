"""Airflow DAG: weekly automated model retraining with validation gate."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "energy-seer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

R2_GATE = 0.70
MIN_SAMPLES = 500


def _fetch_training_data(**ctx) -> dict:
    import os

    from sqlalchemy import create_engine, text

    url = os.getenv("DATABASE_URL", "sqlite:///./energy_seer.db")
    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM energy_readings ORDER BY timestamp DESC LIMIT 5000")
        ).fetchall()

    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"Insufficient data: {len(rows)} rows (need {MIN_SAMPLES})")

    ctx["ti"].xcom_push(key="row_count", value=len(rows))
    print(f"Fetched {len(rows)} rows for retraining")
    return {"row_count": len(rows)}


def _retrain_model(**ctx) -> None:
    import sys

    sys.path.insert(0, "/opt/energy_seer")
    from app.model import generate_synthetic_data, train_model

    X, y = generate_synthetic_data(n=2000)
    _, metrics = train_model(X, y)

    r2 = metrics.get("r2_mean", 0.0)
    if r2 < R2_GATE:
        raise ValueError(f"Model R2={r2:.4f} below gate {R2_GATE}. Aborting.")

    print(f"Retrain complete: R2={r2:.4f} RMSE={metrics.get('rmse_mean', 0):.4f}")
    ctx["ti"].xcom_push(key="r2", value=r2)


def _refresh_anomaly_detector(**ctx) -> None:
    import sys

    sys.path.insert(0, "/opt/energy_seer")
    from app.model import generate_synthetic_data
    from app.monitoring import set_reference_distributions, train_anomaly_detector

    _, y = generate_synthetic_data(n=500)
    train_anomaly_detector(y.tolist())
    set_reference_distributions({"consumption_kwh": y.tolist()})
    print("Anomaly detector refreshed")


def _rebuild_rag_index(**ctx) -> None:
    import sys

    sys.path.insert(0, "/opt/energy_seer")
    from app.model import generate_synthetic_data
    from rag.ingest import ingest_patterns

    X, _ = generate_synthetic_data(n=300)
    patterns = X.to_dict(orient="records")
    count = ingest_patterns(patterns)
    print(f"RAG index rebuilt with {count} patterns")


def _notify_completion(**ctx) -> None:
    r2 = ctx["ti"].xcom_pull(key="r2", task_ids="retrain_model") or 0.0
    print(f"[Energy-Seer] Weekly retrain complete. R2={r2:.4f}")


with DAG(
    dag_id="energy_seer_weekly_retrain",
    default_args=DEFAULT_ARGS,
    description="Weekly automated retraining for Energy-Seer",
    schedule="0 2 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["energy-seer", "ml", "retrain"],
) as dag:
    fetch = PythonOperator(task_id="fetch_training_data", python_callable=_fetch_training_data)
    retrain = PythonOperator(task_id="retrain_model", python_callable=_retrain_model)
    refresh_anomaly = PythonOperator(task_id="refresh_anomaly_detector", python_callable=_refresh_anomaly_detector)
    rebuild_rag = PythonOperator(task_id="rebuild_rag_index", python_callable=_rebuild_rag_index)
    notify = PythonOperator(task_id="notify_completion", python_callable=_notify_completion)

    fetch >> retrain >> [refresh_anomaly, rebuild_rag] >> notify
