"""Airflow DAG for automated model retraining on new delivery data."""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "logistics-flow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _load_training_data(**ctx) -> None:
    """Pull the last 30 days of predictions from PostgreSQL for retraining."""
    import logging
    import os

    import pandas as pd
    from sqlalchemy import create_engine, text

    logger = logging.getLogger(__name__)
    db_url = os.environ["DATABASE_URL"]
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                "SELECT * FROM predictions "
                "WHERE created_at >= NOW() - INTERVAL '30 days'"
            ),
            conn,
        )
    logger.info("Loaded %d training rows", len(df))
    ctx["ti"].xcom_push(key="row_count", value=len(df))
    df.to_parquet("/tmp/retrain_data.parquet", index=False)


def _retrain(**ctx) -> None:
    """Re-fit the ensemble on fresh data if enough rows are available."""
    import logging

    import joblib
    import pandas as pd

    from app.features import build_feature_pipeline, prepare_X
    from app.model import train_model

    logger = logging.getLogger(__name__)
    row_count = ctx["ti"].xcom_pull(key="row_count")
    if row_count < 200:
        logger.warning("Only %d rows — skipping retrain (need >=200)", row_count)
        return

    df = pd.read_parquet("/tmp/retrain_data.parquet")
    feat_pipe = build_feature_pipeline()
    X = prepare_X(df, feat_pipe, fit=True)
    y = df["delivery_minutes"].values
    _, metrics = train_model(X, y)
    joblib.dump(feat_pipe, "feature_pipeline.joblib")
    logger.info("Retrain complete — RMSE=%.2f R²=%.4f", metrics["rmse_mean"], metrics["r2_mean"])


def _check_drift(**ctx) -> None:
    """Run a KS-test drift check and log results."""
    import logging
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.monitoring import run_drift_check

    logger = logging.getLogger(__name__)
    engine = create_engine(os.environ["DATABASE_URL"])
    with Session(engine) as db:
        result = run_drift_check(db)
    drifted = [f for f, r in result.get("features", {}).items() if r.get("drift_detected")]
    if drifted:
        logger.warning("Drift detected in features: %s", drifted)
    else:
        logger.info("No drift detected")


with DAG(
    dag_id="logistics_flow_retrain",
    default_args=DEFAULT_ARGS,
    description="Weekly retraining for Logistics-Flow delivery-time model",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "logistics", "retrain"],
) as dag:
    load_data = PythonOperator(
        task_id="load_training_data",
        python_callable=_load_training_data,
    )
    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=_retrain,
    )
    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=_check_drift,
    )

    load_data >> retrain >> check_drift
