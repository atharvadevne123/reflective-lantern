"""Automated retraining pipeline — Airflow-compatible DAG for Temporal-Pulse."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "reflective-lantern",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2026, 1, 1),
}

SCHEDULE_INTERVAL = "@daily"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./temporal_pulse.db")
LOOKBACK_DAYS = int(os.getenv("RETRAIN_LOOKBACK_DAYS", "30"))
MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "500"))


def extract_recent_readings(lookback_days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Pull recent sensor readings from the database for retraining."""
    from sqlalchemy import create_engine, text

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT sensor_id, timestamp, values FROM sensor_readings "
                    "WHERE timestamp >= :cutoff ORDER BY timestamp"
                ),
                {"cutoff": cutoff},
            ).fetchall()
        readings = [
            {"sensor_id": r[0], "timestamp": r[1], **(r[2] if isinstance(r[2], dict) else {})}
            for r in rows
        ]
        logger.info("Extracted %d readings since %s", len(readings), cutoff.date())
        return readings
    except Exception as e:
        logger.error("Failed to extract readings: %s", e)
        return []


def run_feature_engineering(readings: list[dict[str, Any]]) -> tuple[Any, list[str]]:
    """Run the feature engineering pipeline on extracted readings."""
    import numpy as np
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.features import build_feature_matrix

    df, feature_cols = build_feature_matrix(readings)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    logger.info("Feature matrix: %d samples × %d features", *X.shape)
    return X, feature_cols


def train_and_evaluate(X: Any, contamination: float = 0.05) -> dict[str, Any]:
    """Train anomaly detector and forecaster and return metrics."""
    import numpy as np
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.model import train_anomaly_detector, train_forecaster, get_feature_importance

    if_model, scaler = train_anomaly_detector(X, contamination=contamination)
    y = X[:, 0]
    rf_model, metrics = train_forecaster(X, y)
    importances = get_feature_importance()
    metrics["top_features"] = importances[:5]
    return metrics


def update_reference_distributions(X: Any, feature_cols: list[str]) -> None:
    """Update the reference distributions used for KS drift testing."""
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from app.monitoring import update_reference_distribution

    for idx, col in enumerate(feature_cols[:20]):
        update_reference_distribution(col, X[:, idx].tolist())
    logger.info("Reference distributions updated for %d features", min(20, len(feature_cols)))


def run_pipeline(contamination: float = 0.05) -> dict[str, Any]:
    """Execute the full retraining pipeline end-to-end.

    This function serves as both the Airflow entrypoint and the
    standalone execution path.
    """
    logger.info("Starting Temporal-Pulse retraining pipeline")
    readings = extract_recent_readings()
    if len(readings) < MIN_SAMPLES:
        logger.warning(
            "Only %d readings available (min=%d); skipping retraining",
            len(readings),
            MIN_SAMPLES,
        )
        return {"status": "skipped", "reason": "insufficient_data", "n_readings": len(readings)}

    try:
        X, feature_cols = run_feature_engineering(readings)
        metrics = train_and_evaluate(X, contamination=contamination)
        update_reference_distributions(X, feature_cols)
        result = {
            "status": "success",
            "n_readings": len(readings),
            "n_features": len(feature_cols),
            "metrics": metrics,
        }
        logger.info("Retraining pipeline completed: %s", result)
        return result
    except Exception as e:
        logger.error("Retraining pipeline failed: %s", e)
        return {"status": "failed", "error": str(e)}


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="temporal_pulse_retrain",
        default_args=DEFAULT_ARGS,
        schedule_interval=SCHEDULE_INTERVAL,
        catchup=False,
        tags=["temporal-pulse", "ml", "retraining"],
    ) as dag:
        extract_task = PythonOperator(
            task_id="extract_readings",
            python_callable=extract_recent_readings,
        )
        train_task = PythonOperator(
            task_id="train_and_evaluate",
            python_callable=run_pipeline,
        )
        extract_task >> train_task

except ImportError:
    logger.debug("Airflow not installed; DAG definition skipped")

if __name__ == "__main__":
    import json

    result = run_pipeline()
    print(json.dumps(result, indent=2, default=str))
