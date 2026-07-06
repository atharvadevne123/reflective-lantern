"""Airflow DAG for automated weekly model retraining."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "realty-edge",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

MIN_SAMPLES = 200
MIN_R2 = 0.70


def _fetch_training_data(**context: object) -> dict:
    import json
    import os

    from sqlalchemy import create_engine, text

    db_url = os.getenv("DATABASE_URL", "sqlite:///./realty_edge.db")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT features_json, predicted_value FROM prediction_logs ORDER BY created_at DESC LIMIT 5000")).fetchall()
    print(f"Fetched {len(rows)} rows")
    if len(rows) < MIN_SAMPLES:
        raise ValueError(f"Insufficient data: {len(rows)} < {MIN_SAMPLES}")
    records = []
    for row in rows:
        try:
            feat = json.loads(row[0])
            feat["_target"] = row[1]
            records.append(feat)
        except Exception:
            pass
    context["ti"].xcom_push(key="record_count", value=len(records))
    return {"records": records}


def _train_and_validate(**context: object) -> None:
    import numpy as np
    import pandas as pd
    from app.model import train_model

    records = context["ti"].xcom_pull(task_ids="fetch_data", key="records") or []
    if not records:
        raise ValueError("No records received from XCom")
    df = pd.DataFrame(records)
    target_col = "_target"
    if target_col not in df.columns:
        raise ValueError("Missing target column")
    y = df.pop(target_col).values.astype(np.float32)
    _, metrics = train_model(df, y)
    print(f"Retrain metrics: {metrics}")
    if metrics.get("r2_mean", 0) < MIN_R2:
        raise ValueError(f"R2 {metrics['r2_mean']:.4f} below threshold {MIN_R2}")
    context["ti"].xcom_push(key="r2_mean", value=metrics["r2_mean"])


def _run_drift_check(**context: object) -> None:
    import os

    from app.monitoring import get_recent_predictions, run_drift_check
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_url = os.getenv("DATABASE_URL", "sqlite:///./realty_edge.db")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        preds = get_recent_predictions(db, limit=500)
        vals = [p.predicted_value for p in preds]
        if len(vals) >= 40:
            mid = len(vals) // 2
            report = run_drift_check(db, "predicted_value", vals[:mid], vals[mid:])
            print(f"Drift check: ks={report.ks_statistic} p={report.p_value} detected={report.drift_detected}")
    finally:
        db.close()


with DAG(
    dag_id="realty_edge_weekly_retrain",
    default_args=DEFAULT_ARGS,
    description="Weekly model retraining and drift check for Realty-Edge",
    schedule="0 2 * * 1",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["realty-edge", "ml", "retrain"],
) as dag:
    fetch_data = PythonOperator(task_id="fetch_data", python_callable=_fetch_training_data)
    train_validate = PythonOperator(task_id="train_validate", python_callable=_train_and_validate)
    drift_check = PythonOperator(task_id="drift_check", python_callable=_run_drift_check)

    fetch_data >> train_validate >> drift_check
