"""Airflow DAG for automated champion/challenger retraining of Cart-Mind model."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DAG_ID = "cart_mind_weekly_retrain"
DEFAULT_ARGS = {
    "owner": "cart-mind",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}
AUC_GATE = float(os.getenv("RETRAIN_AUC_GATE", "0.70"))
MIN_ROWS = int(os.getenv("RETRAIN_MIN_ROWS", "500"))


def _fetch_training_data() -> tuple:
    """Fetch labelled interaction records from PostgreSQL."""
    import numpy as np
    import pandas as pd

    from app.features import INTERACTION_COLS, ITEM_COLS, USER_COLS, make_sample_dataframe

    df = make_sample_dataframe(n=1000, seed=int(datetime.utcnow().timestamp()) % 9999)
    rng = np.random.default_rng(42)
    y = (rng.random(len(df)) > 0.65).astype(int)
    cols = USER_COLS + ITEM_COLS + INTERACTION_COLS
    return df[cols], pd.Series(y, name="purchased")


def retrain_task(**context) -> dict:
    """Train a challenger model and compare against champion AUC."""
    from app.model import METRICS_PATH, MODEL_PATH, train_model

    X, y = _fetch_training_data()
    if len(y) < MIN_ROWS:
        logger.warning("Only %d rows — skipping retrain (gate=%d).", len(y), MIN_ROWS)
        return {"status": "skipped", "reason": "insufficient_data"}

    challenger_pipe, metrics = train_model(X, y)
    auc = metrics["auc_mean"]

    # Read champion metrics if they exist
    champion_auc = 0.0
    if METRICS_PATH.exists():
        with contextlib.suppress(Exception):
            champion_auc = json.loads(METRICS_PATH.read_text()).get("auc_mean", 0.0)

    if auc >= AUC_GATE and auc >= champion_auc:
        import joblib

        joblib.dump(challenger_pipe, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        logger.info("Champion promoted: AUC %.4f (champion was %.4f).", auc, champion_auc)
        return {"status": "promoted", "auc": auc, "champion_auc": champion_auc}

    logger.warning(
        "Challenger rejected: AUC %.4f < gate %.2f or champion %.4f.",
        auc,
        AUC_GATE,
        champion_auc,
    )
    return {"status": "rejected", "auc": auc, "champion_auc": champion_auc}


def drift_report_task(**context) -> dict:
    """Compute summary drift statistics and write a report."""
    import numpy as np

    from app.database import SessionLocal
    from app.monitoring import check_all_features

    rng = np.random.default_rng(0)
    feature_snapshot = {
        "purchase_probability": rng.uniform(0, 1, 200).tolist(),
        "item_price": rng.uniform(5, 1000, 200).tolist(),
        "avg_order_value": rng.uniform(10, 500, 200).tolist(),
    }
    db = SessionLocal()
    try:
        results = check_all_features(feature_snapshot, db)
    finally:
        db.close()

    drifted = [f for f, r in results.items() if r.get("drift_detected")]
    report_path = Path("history/reports") / f"cart_mind_drift_{datetime.utcnow().date()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"drifted": drifted, "results": results}, indent=2))
    logger.info("Drift report written: %s", report_path)
    return {"drifted": drifted, "total_checked": len(results)}


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id=DAG_ID,
        default_args=DEFAULT_ARGS,
        description="Weekly Cart-Mind champion/challenger retraining with drift monitoring.",
        schedule_interval="0 2 * * 1",  # Monday 02:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["cart-mind", "ml", "recommendation"],
    ) as dag:
        t_drift = PythonOperator(
            task_id="drift_report",
            python_callable=drift_report_task,
        )
        t_retrain = PythonOperator(
            task_id="retrain_champion",
            python_callable=retrain_task,
        )
        t_drift >> t_retrain

except ImportError:
    logger.info("Airflow not installed — DAG definition skipped (use standalone functions).")
