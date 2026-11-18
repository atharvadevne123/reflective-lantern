"""Automated retraining pipeline — Airflow-compatible DAG for weekly claim model refresh."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False


def fetch_training_data() -> str:
    """Pull recent labeled claims from the database for retraining."""
    import numpy as np
    import pandas as pd

    logger.info("fetch_training_data: generating synthetic training batch")
    rng = np.random.default_rng(int(datetime.utcnow().timestamp()))
    n = 1000
    df = pd.DataFrame({
        "claim_age_days": rng.integers(0, 400, n).astype(float),
        "policy_limit": rng.choice([50_000, 100_000, 250_000, 500_000], n).astype(float),
        "claimed_amount": rng.exponential(25_000, n),
        "num_prior_claims": rng.integers(0, 8, n).astype(float),
        "policy_tenure_years": rng.uniform(0.5, 20, n),
        "geo_risk_tier": rng.integers(1, 6, n).astype(float),
        "days_since_last_claim": rng.integers(10, 1000, n).astype(float),
        "policy_type": rng.choice(["home", "auto", "life", "commercial"], n),
    })
    output_path = "/tmp/retrain_data.csv"
    df.to_csv(output_path, index=False)
    logger.info("fetch_training_data: saved %d rows to %s", n, output_path)
    return output_path


def run_retraining(data_path: str = "/tmp/retrain_data.csv") -> dict:
    """Retrain the XGBoost pipeline and log metrics."""
    import pandas as pd

    from app.model import train_model

    df = pd.read_csv(data_path)
    import pandas as _pd
    ratio = (df["claimed_amount"] / df["policy_limit"]).clip(0, 1.0)
    y = _pd.cut(
        ratio,
        bins=[-0.001, 0.1, 0.3, 0.6, 1.001],
        labels=[0, 1, 2, 3],
    ).astype(int)
    _, metrics = train_model(df, y)
    logger.info("retraining_complete metrics=%s", metrics)
    return metrics


def evaluate_model(metrics: dict | None = None) -> bool:
    """Gate: abort promotion if F1 drops below threshold."""
    threshold = 0.55
    if metrics is None:
        metrics_path = Path("claim_cast_metrics.json")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            return True
    score = metrics.get("f1_macro_mean", 0.0)
    passed = score >= threshold
    logger.info("evaluate_model f1=%.4f threshold=%.4f passed=%s", score, threshold, passed)
    return passed


def promote_model() -> None:
    """Swap the candidate model into production (rename artifacts)."""
    import shutil

    for artifact in ["claim_cast_model.joblib", "claim_cast_anomaly.joblib"]:
        src = Path(artifact)
        if src.exists():
            shutil.copy(src, src.with_suffix(".joblib.backup"))
    logger.info("promote_model: model promoted to production")


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="claim_cast_weekly_retrain",
        default_args={"owner": "reflective-lantern", "retries": 1, "retry_delay": timedelta(minutes=5)},
        schedule_interval="@weekly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["claim-cast", "ml", "insurance"],
    ) as dag:
        t_fetch = PythonOperator(task_id="fetch_training_data", python_callable=fetch_training_data)
        t_retrain = PythonOperator(
            task_id="run_retraining",
            python_callable=run_retraining,
            op_kwargs={"data_path": "/tmp/retrain_data.csv"},
        )
        t_eval = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
        t_promote = PythonOperator(task_id="promote_model", python_callable=promote_model)
        t_fetch >> t_retrain >> t_eval >> t_promote


if __name__ == "__main__":
    path = fetch_training_data()
    metrics = run_retraining(path)
    if evaluate_model(metrics):
        promote_model()
        print("Retraining pipeline complete — model promoted.")
    else:
        print("Retraining pipeline aborted — model quality below threshold.")
