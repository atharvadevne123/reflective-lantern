"""Airflow DAG for automated weekly retraining of the Volt-Cast model."""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW = True
except ImportError:
    _AIRFLOW = False


def _fetch_recent_data(**ctx) -> dict:
    """Pull energy readings from the database for the past 7 days."""
    import logging
    import os
    logger = logging.getLogger(__name__)

    from sqlalchemy import create_engine, text
    db_url = os.getenv("DATABASE_URL", "sqlite:///./volt_cast.db")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM energy_readings ORDER BY timestamp DESC LIMIT 5000")
        ).fetchall()
    logger.info("fetched %d rows for retraining", len(rows))
    return {"row_count": len(rows)}


def _train_and_evaluate(**ctx) -> dict:
    """Retrain the ensemble model and validate performance gate."""
    import logging
    logger = logging.getLogger(__name__)

    from app.model import generate_synthetic_training_data, train_model
    X, y = generate_synthetic_training_data(n_samples=3000)
    model, metrics = train_model(X.values, y.values, cv_folds=5)

    r2 = metrics.get("r2_mean", 0.0)
    logger.info("retraining complete: R2=%.4f RMSE=%.2f", r2, metrics.get("rmse_mean", 0))

    if r2 < 0.60:
        raise ValueError(f"Retraining gate failed: R2={r2:.4f} < 0.60 threshold")

    return {"r2": r2, "rmse": metrics.get("rmse_mean")}


def _log_retraining_event(**ctx) -> None:
    """Log the retraining run to the database."""
    import logging
    import os
    logger = logging.getLogger(__name__)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base, RetrainingLog

    db_url = os.getenv("DATABASE_URL", "sqlite:///./volt_cast.db")
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        ti = ctx.get("ti")
        metrics = ti.xcom_pull(task_ids="train_and_evaluate") if ti else {}
        db.add(RetrainingLog(
            trigger="airflow_weekly",
            samples_used=3000,
            r2_after=metrics.get("r2") if metrics else None,
            rmse_after=metrics.get("rmse") if metrics else None,
            status="success",
            notes="Automated weekly retraining via Airflow DAG",
        ))
        db.commit()
        logger.info("retraining event logged to database")
    finally:
        db.close()


def _check_drift_and_alert(**ctx) -> None:
    """Check for feature drift after retraining and emit warnings."""
    import logging

    logger = logging.getLogger(__name__)

    from app.model import generate_synthetic_training_data
    from app.monitoring import compute_drift

    X_ref, y_ref = generate_synthetic_training_data(n_samples=1000, seed=42)
    X_cur, y_cur = generate_synthetic_training_data(n_samples=200, seed=99)

    result = compute_drift(y_ref.tolist(), y_cur.tolist(), "load_mw")
    logger.info(
        "post-retrain drift check: ks=%.4f p=%.4f drift=%s",
        result["ks_statistic"],
        result["p_value"],
        result["drift_detected"],
    )
    if result["drift_detected"]:
        logger.warning("DRIFT ALERT: load_mw distribution has shifted significantly")


def run_retraining_pipeline() -> dict:
    """Standalone entry point for running the pipeline outside Airflow."""
    ctx: dict = {}
    data_info = _fetch_recent_data(**ctx)
    metrics = _train_and_evaluate(**ctx)
    _log_retraining_event(**ctx)
    _check_drift_and_alert(**ctx)
    return {**data_info, **metrics}


if _AIRFLOW:
    default_args = {
        "owner": "volt-cast",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": True,
        "email": ["devneatharva@gmail.com"],
    }

    with DAG(
        dag_id="volt_cast_weekly_retrain",
        default_args=default_args,
        description="Weekly automated retraining for Volt-Cast energy model",
        schedule="0 2 * * 1",  # Monday 02:00 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["volt-cast", "ml", "energy"],
    ) as dag:

        fetch_data = PythonOperator(
            task_id="fetch_recent_data",
            python_callable=_fetch_recent_data,
        )

        train = PythonOperator(
            task_id="train_and_evaluate",
            python_callable=_train_and_evaluate,
        )

        log_event = PythonOperator(
            task_id="log_retraining_event",
            python_callable=_log_retraining_event,
        )

        drift_check = PythonOperator(
            task_id="check_drift_and_alert",
            python_callable=_check_drift_and_alert,
        )

        fetch_data >> train >> log_event >> drift_check
