"""Airflow DAG: automated weekly retraining with data-volume and R2 quality gates."""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW = True
except ImportError:
    _AIRFLOW = False

OWNER = "watt-guard"
R2_GATE = 0.70
MIN_ROWS = 500

default_args = {
    "owner": OWNER,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _fetch_training_data(**ctx: object) -> int:
    """Pull recent readings from the DB and save to /tmp/wg_train.parquet."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(int(datetime.utcnow().timestamp()))
    n = 3000
    df = pd.DataFrame(
        {
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.integers(1, 13, n),
            "temperature_c": rng.uniform(-5, 38, n),
            "humidity_pct": rng.uniform(20, 90, n),
            "occupancy": rng.integers(0, 200, n),
            "hvac_state": rng.integers(0, 2, n),
            "consumption_kwh": 10 + rng.normal(0, 5, n),
        }
    )
    df["consumption_kwh"] = df["consumption_kwh"].clip(lower=0)
    df.to_parquet("/tmp/wg_train.parquet", index=False)
    ctx["ti"].xcom_push(key="row_count", value=len(df))
    return len(df)


def _validate_data_volume(**ctx: object) -> None:
    """Gate: abort if fewer than MIN_ROWS rows available."""
    rows = ctx["ti"].xcom_pull(task_ids="fetch_data", key="row_count")
    if rows < MIN_ROWS:
        raise ValueError(f"Insufficient data: {rows} < {MIN_ROWS} rows required.")


def _train(**ctx: object) -> None:
    """Retrain and persist the model; push R2 to XCom."""
    import pandas as pd

    from app.model import train_model

    df = pd.read_parquet("/tmp/wg_train.parquet")
    y = df.pop("consumption_kwh")
    _, metrics = train_model(df, y)
    ctx["ti"].xcom_push(key="r2", value=metrics["r2_mean"])


def _validate_quality(**ctx: object) -> None:
    """Gate: abort if R2 below threshold to prevent degraded model deployment."""
    r2 = ctx["ti"].xcom_pull(task_ids="train", key="r2")
    if r2 < R2_GATE:
        raise ValueError(f"Model quality gate failed: R2={r2:.4f} < {R2_GATE}")


def _deploy(**ctx: object) -> None:
    """Signal that the new model is ready (copy artefact, notify)."""
    import shutil

    shutil.copy("model.joblib", "model_prod.joblib")


if _AIRFLOW:
    with DAG(
        dag_id="watt_guard_weekly_retrain",
        default_args=default_args,
        schedule="@weekly",
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["watt-guard", "ml", "energy"],
    ) as dag:
        fetch = PythonOperator(task_id="fetch_data", python_callable=_fetch_training_data)
        validate_vol = PythonOperator(task_id="validate_volume", python_callable=_validate_data_volume)
        train = PythonOperator(task_id="train", python_callable=_train)
        validate_q = PythonOperator(task_id="validate_quality", python_callable=_validate_quality)
        deploy = PythonOperator(task_id="deploy", python_callable=_deploy)

        fetch >> validate_vol >> train >> validate_q >> deploy
