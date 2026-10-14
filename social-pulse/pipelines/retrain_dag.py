"""Airflow DAG for automated model retraining when drift is detected."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG  # type: ignore[import]
from airflow.operators.python import PythonOperator  # type: ignore[import]


logger = logging.getLogger(__name__)

DEFAULT_ARGS: dict = {
    "owner": "social-pulse",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def check_drift_task(**context: dict) -> bool:
    """Check if drift has been detected in the last 7 days.

    Returns True if retraining is required; pushes to XCom.
    """
    import os
    import sys

    sys.path.insert(0, "/opt/social-pulse")

    from app.database import SessionLocal, init_db
    from app.monitoring import run_drift_check, should_retrain

    init_db()
    db = SessionLocal()
    try:
        records = run_drift_check(db, reference_window_days=30, current_window_days=7)
        drifted_features = [r.feature_name for r in records if r.drift_detected]
        needs_retrain = should_retrain(db) or len(drifted_features) > 0
        context["task_instance"].xcom_push(key="needs_retrain", value=needs_retrain)
        context["task_instance"].xcom_push(key="drifted_features", value=drifted_features)
        logger.info(
            "Drift check: %d features drifted, needs_retrain=%s",
            len(drifted_features),
            needs_retrain,
        )
        return needs_retrain
    finally:
        db.close()


def retrain_model_task(**context: dict) -> dict:
    """Retrain model if drift was detected; otherwise skip.

    Returns a dict with training metrics pushed to XCom.
    """
    import sys

    sys.path.insert(0, "/opt/social-pulse")

    needs_retrain: bool = context["task_instance"].xcom_pull(
        task_ids="check_drift", key="needs_retrain"
    )
    if not needs_retrain:
        logger.info("No drift detected — skipping retraining")
        return {"skipped": True}

    from app.database import SessionLocal, init_db
    from app.model import generate_synthetic_data, load_model, train

    init_db()
    db = SessionLocal()
    try:
        from app.database import Prediction

        rows = db.query(Prediction).all()
        if rows:
            samples = [
                {
                    "text": r.text,
                    "author_followers": r.author_followers,
                    "has_media": r.has_media,
                    "has_link": r.has_link,
                    "hashtag_count": r.hashtag_count,
                    "mention_count": r.mention_count,
                    "hour_of_day": r.hour_of_day,
                    "day_of_week": r.day_of_week,
                }
                for r in rows
            ]
            labels = [int(r.viral_label) for r in rows]
            logger.info("Retraining on %d real samples", len(samples))
        else:
            samples, labels = generate_synthetic_data(n=500)
            logger.info("No real data; retraining on 500 synthetic samples")

        result = train(samples, labels)
        metrics = {
            "run_id": result.run_id,
            "auc_roc": result.auc_roc,
            "f1": result.f1,
            "train_samples": result.train_samples,
        }
        context["task_instance"].xcom_push(key="train_metrics", value=metrics)
        logger.info("Retraining complete: AUC=%.4f", result.auc_roc)
        return metrics
    finally:
        db.close()


def notify_task(**context: dict) -> None:
    """Log a summary of the retraining run."""
    needs_retrain: bool = context["task_instance"].xcom_pull(
        task_ids="check_drift", key="needs_retrain"
    )
    if not needs_retrain:
        logger.info("Pipeline complete: no retraining needed")
        return
    metrics: dict = context["task_instance"].xcom_pull(
        task_ids="retrain_model", key="train_metrics"
    ) or {}
    logger.info(
        "Pipeline complete: run_id=%s AUC=%.4f samples=%s",
        metrics.get("run_id"),
        metrics.get("auc_roc", 0),
        metrics.get("train_samples", 0),
    )


with DAG(
    dag_id="social_pulse_retrain",
    default_args=DEFAULT_ARGS,
    description="Weekly drift check and model retraining for Social-Pulse",
    schedule_interval="0 2 * * 0",  # every Sunday at 02:00 UTC
    catchup=False,
    tags=["ml", "social-pulse"],
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift_task,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=notify_task,
    )

    check_drift >> retrain_model >> notify
