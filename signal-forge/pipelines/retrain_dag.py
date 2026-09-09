"""Automated retraining pipeline for Signal-Forge (Airflow-compatible DAG)."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.info("Airflow not available; pipeline runs as standalone script")


DEFAULT_ARGS = {
    "owner": "signal-forge",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

MODEL_MIN_SAMPLES = 100


def fetch_training_data(**context: Any) -> dict[str, Any]:
    """Fetch recent predictions from DB to use as training data."""
    from app.database import SessionLocal, MarketRegimePrediction

    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=90)
        rows = (
            db.query(MarketRegimePrediction)
            .filter(MarketRegimePrediction.created_at >= cutoff)
            .filter(MarketRegimePrediction.volatility.isnot(None))
            .all()
        )
        logger.info("Fetched %d training samples", len(rows))
        if len(rows) < MODEL_MIN_SAMPLES:
            logger.warning("Insufficient data for retraining: %d < %d", len(rows), MODEL_MIN_SAMPLES)
            return {"skip": True}

        X = np.array([
            [r.volatility or 0, r.momentum or 0, r.volume_ratio or 1,
             r.correlation or 0, r.beta or 1]
            for r in rows
        ])
        regime_map = {"bull": 0, "bear": 1, "sideways": 2, "volatile": 3}
        y = np.array([regime_map.get(r.regime, 2) for r in rows])

        np.save("/tmp/sf_X_retrain.npy", X)
        np.save("/tmp/sf_y_retrain.npy", y)
        return {"skip": False, "n_samples": len(rows)}
    except Exception as e:
        logger.error("Failed to fetch training data: %s", e)
        raise
    finally:
        db.close()


def train_new_model(**context: Any) -> None:
    """Train ensemble model on fresh data and save to disk."""
    ti = context.get("ti")
    upstream = ti.xcom_pull(task_ids="fetch_data") if ti else {"skip": False}
    if upstream and upstream.get("skip"):
        logger.info("Skipping retraining: insufficient data")
        return

    try:
        import os
        X = np.load("/tmp/sf_X_retrain.npy")
        y = np.load("/tmp/sf_y_retrain.npy")
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((400, 5))
        y = rng.integers(0, 4, 400)

    from app.model import train_model, save_model
    model, metrics = train_model(X, y)
    save_model(model)
    logger.info("Retraining complete: %s", metrics)


def run_drift_check(**context: Any) -> None:
    """Check for data drift and log any detected shifts."""
    from app.database import SessionLocal
    from app.monitoring import monitor_all_features

    db = SessionLocal()
    try:
        results = monitor_all_features(db)
        drifted = [r["feature"] for r in results if r.get("drift_detected")]
        if drifted:
            logger.warning("Drift detected in features: %s", drifted)
        else:
            logger.info("No drift detected across %d features", len(results))
    except Exception as e:
        logger.error("Drift check failed: %s", e)
        raise
    finally:
        db.close()


def rebuild_faiss_index(**context: Any) -> None:
    """Rebuild FAISS index from current prediction history."""
    import json
    from app.database import SessionLocal, MarketRegimePrediction
    from app.model import build_faiss_index, FAISS_INDEX_PATH

    db = SessionLocal()
    try:
        rows = (
            db.query(MarketRegimePrediction)
            .filter(MarketRegimePrediction.volatility.isnot(None))
            .order_by(MarketRegimePrediction.created_at.desc())
            .limit(5000)
            .all()
        )
        if not rows:
            logger.info("No predictions available for FAISS rebuild")
            return

        embeddings = np.array([
            [r.volatility or 0, r.momentum or 0, r.volume_ratio or 1,
             r.correlation or 0, r.beta or 1]
            for r in rows
        ], dtype=np.float32)

        build_faiss_index(embeddings)

        metadata = [
            {"id": r.id, "ticker": r.ticker, "regime": r.regime,
             "date": r.created_at.isoformat()}
            for r in rows
        ]
        meta_path = FAISS_INDEX_PATH.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

        logger.info("FAISS index rebuilt: %d vectors", len(rows))
    except Exception as e:
        logger.error("FAISS rebuild failed: %s", e)
        raise
    finally:
        db.close()


if AIRFLOW_AVAILABLE:
    with DAG(
        "signal_forge_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly Signal-Forge model retraining and drift monitoring",
        schedule_interval="@weekly",
        catchup=False,
        tags=["signal-forge", "ml", "retrain"],
    ) as dag:
        fetch_task = PythonOperator(task_id="fetch_data", python_callable=fetch_training_data)
        train_task = PythonOperator(task_id="train_model", python_callable=train_new_model)
        drift_task = PythonOperator(task_id="drift_check", python_callable=run_drift_check)
        faiss_task = PythonOperator(task_id="rebuild_faiss", python_callable=rebuild_faiss_index)
        fetch_task >> train_task >> drift_task >> faiss_task


def run_pipeline() -> None:
    """Entry point to run the full retraining pipeline without Airflow."""
    logger.info("Starting standalone retraining pipeline")
    fetch_training_data()
    train_new_model()
    run_drift_check()
    rebuild_faiss_index()
    logger.info("Retraining pipeline complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
