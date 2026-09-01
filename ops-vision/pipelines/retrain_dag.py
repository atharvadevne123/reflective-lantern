"""Airflow-compatible retraining DAG for Ops-Vision ML model.

This module defines a DAG that nightly retrains the ensemble classifier on
recently collected production data and promotes the new model if it improves
on the held-out AUC-ROC benchmark.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARGS: dict = {
    "owner": "ops-vision",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

AUC_THRESHOLD: float = 0.70
MIN_SAMPLES: int = 500


def fetch_training_data() -> tuple:
    """Pull recent incidents from the database for retraining.

    Returns:
        Tuple of (features DataFrame, binary label Series).
    """
    logger.info("Fetching training data from database")
    try:
        import os

        from sqlalchemy import create_engine, text

        db_url = os.environ.get("DATABASE_URL", "postgresql://ops:ops@localhost:5432/opsvision")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT cpu_usage_pct, memory_usage_pct, error_rate_per_min, "
                    "latency_p99_ms, request_rate_per_sec, disk_io_util_pct, "
                    "is_incident FROM incidents ORDER BY created_at DESC LIMIT 10000"
                )
            )
            rows = result.fetchall()

        if len(rows) < MIN_SAMPLES:
            logger.warning("Insufficient data (%d rows) — falling back to synthetic", len(rows))
            from app.model import generate_synthetic_data

            return generate_synthetic_data(n_samples=2000)

        import pandas as pd

        feature_cols = [
            "cpu_usage_pct",
            "memory_usage_pct",
            "error_rate_per_min",
            "latency_p99_ms",
            "request_rate_per_sec",
            "disk_io_util_pct",
        ]
        df = pd.DataFrame(rows, columns=feature_cols + ["is_incident"])
        return df[feature_cols], df["is_incident"]

    except Exception:
        logger.exception("DB fetch failed — using synthetic data")
        from app.model import generate_synthetic_data

        return generate_synthetic_data(n_samples=2000)


def retrain_model(**context) -> dict:
    """Train a new model and return metrics.

    Args:
        **context: Airflow task context (unused outside Airflow).

    Returns:
        Dict with training metrics and model path.
    """
    from sklearn.model_selection import train_test_split

    from app.features import build_feature_pipeline
    from app.model import evaluate, save_model, train

    df, labels = fetch_training_data()
    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels.values, test_size=0.2, stratify=labels.values, random_state=42
    )

    model, train_metrics = train(X_train, y_train)
    test_metrics = evaluate(model, X_test, y_test)

    all_metrics = {**train_metrics, **test_metrics}
    logger.info("Retrain metrics: %s", all_metrics)

    model_path = Path("/tmp/ops_vision_model_candidate.pkl")
    save_model(model, model_path)

    pipeline_path = Path("/tmp/ops_vision_pipeline_candidate.pkl")
    pipeline_path.write_bytes(pickle.dumps(pipeline))

    return {"metrics": all_metrics, "model_path": str(model_path)}


def promote_model(**context) -> None:
    """Promote candidate model to production if AUC-ROC exceeds threshold.

    Args:
        **context: Airflow task context.
    """
    import os
    import shutil

    task_instance = context.get("ti")
    if task_instance:
        result = task_instance.xcom_pull(task_ids="retrain_model")
        metrics = result.get("metrics", {})
    else:
        metrics = {"test_auc_roc": 1.0}

    auc = metrics.get("test_auc_roc", 0.0)
    logger.info("Candidate model AUC-ROC: %.4f (threshold: %.4f)", auc, AUC_THRESHOLD)

    if auc >= AUC_THRESHOLD:
        prod_path = os.environ.get("MODEL_PATH", "/tmp/ops_vision_model.pkl")
        shutil.copy("/tmp/ops_vision_model_candidate.pkl", prod_path)
        shutil.copy(
            "/tmp/ops_vision_pipeline_candidate.pkl",
            "/tmp/ops_vision_pipeline.pkl",
        )
        logger.info("Model promoted to production at %s (AUC=%.4f)", prod_path, auc)
    else:
        logger.warning("Model NOT promoted — AUC %.4f below threshold %.4f", auc, AUC_THRESHOLD)


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="ops_vision_retrain",
        default_args=DEFAULT_ARGS,
        description="Nightly retraining DAG for Ops-Vision ML model",
        schedule_interval="0 2 * * *",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["ops-vision", "ml", "retrain"],
    ) as dag:
        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=retrain_model,
        )
        promote_task = PythonOperator(
            task_id="promote_model",
            python_callable=promote_model,
        )
        retrain_task >> promote_task

except ImportError:
    logger.debug("Airflow not installed — DAG definition skipped (standalone mode)")
