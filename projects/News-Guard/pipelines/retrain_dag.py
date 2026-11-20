"""Automated retraining pipeline (Airflow DAG) for News-Guard."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False

DEFAULT_ARGS = {
    "owner": "news-guard",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _check_drift(**ctx) -> bool:
    """Check KS-drift against reference distribution. Push XCom result."""
    from app.database import SessionLocal
    from app.model import NewsGuardModel
    from app.monitoring import compute_drift, get_recent_predictions

    db = SessionLocal()
    try:
        model: NewsGuardModel = NewsGuardModel.load()
        ref = model.get_reference_features()
        recent = get_recent_predictions(db, hours=168)
        result = compute_drift(ref, recent)
        logger.info("Drift check result: %s", result)
        ctx["ti"].xcom_push(key="drift_result", value=result)
        return result["drift_detected"]
    finally:
        db.close()


def _retrain_model(**ctx) -> None:
    """Retrain model on fresh synthetic data (replace with real DB query in prod)."""
    from app.model import NewsGuardModel, generate_synthetic_data

    drift = ctx["ti"].xcom_pull(key="drift_result", task_ids="check_drift")
    if drift and not drift.get("drift_detected", False):
        logger.info("No drift detected; skipping retrain.")
        return

    logger.info("Retraining model...")
    df, y = generate_synthetic_data(n_samples=3000)
    model = NewsGuardModel()
    metrics = model.train(df, y)
    logger.info("Retrain complete. AUC=%.4f", metrics["auc_mean"])


def _validate_model(**ctx) -> None:
    """Validate newly trained model meets AUC threshold."""
    import json
    from pathlib import Path
    metrics = json.loads(Path("metrics.json").read_text())
    auc = metrics.get("auc_mean", 0.0)
    if auc < 0.75:
        raise ValueError(f"Model AUC {auc:.4f} below threshold 0.75; rejecting.")
    logger.info("Model validation passed. AUC=%.4f", auc)


def _notify_completion(**ctx) -> None:
    logger.info("Retraining pipeline completed at %s", datetime.utcnow().isoformat())


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="news_guard_retrain",
        default_args=DEFAULT_ARGS,
        description="Automated retraining for News-Guard on drift detection",
        schedule_interval="0 2 * * 0",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["news-guard", "ml", "retraining"],
    ) as dag:
        check_drift = PythonOperator(task_id="check_drift", python_callable=_check_drift)
        retrain = PythonOperator(task_id="retrain_model", python_callable=_retrain_model)
        validate = PythonOperator(task_id="validate_model", python_callable=_validate_model)
        notify = PythonOperator(task_id="notify_completion", python_callable=_notify_completion)

        check_drift >> retrain >> validate >> notify
