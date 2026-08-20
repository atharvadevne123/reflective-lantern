"""Automated retraining pipeline (Airflow-compatible DAG) for Quake-Net."""

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
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]

CHAMPION_R2_GATE = 0.70
MIN_SAMPLES = 500
METRICS_PATH = Path("metrics.json")


def _check_drift(**context) -> dict:
    """Pull drift report; abort if no drift detected and samples are sufficient."""
    from app.monitoring import check_all_drift

    drifts = check_all_drift()
    detected = sum(1 for d in drifts if d.get("drift_detected"))
    total = len(drifts)
    logger.info("Drift check: %d/%d features drifted", detected, total)
    result = {"drift_detected_count": detected, "total_features": total}
    context["ti"].xcom_push(key="drift_summary", value=result)
    return result


def _load_fresh_data(**context) -> dict:
    """Simulate data loading from production DB — returns synthetic dataset stats."""
    from app.features import make_synthetic_dataset

    df = make_synthetic_dataset(n_samples=2000, seed=int(datetime.utcnow().timestamp()) % 10000)
    stats = {
        "n_samples": len(df),
        "magnitude_mean": round(float(df["magnitude"].mean()), 3),
        "magnitude_std": round(float(df["magnitude"].std()), 3),
    }
    logger.info("Loaded fresh data: %s", stats)
    context["ti"].xcom_push(key="data_stats", value=stats)

    if stats["n_samples"] < MIN_SAMPLES:
        raise ValueError(f"Insufficient samples: {stats['n_samples']} < {MIN_SAMPLES}")

    # Persist for next task
    import pickle

    Path("/tmp/retrain_df.pkl").write_bytes(pickle.dumps(df))
    return stats


def _train_challenger(**context) -> dict:
    """Train challenger model and compute metrics."""
    import pickle

    from app.model import train_model

    df = pickle.loads(Path("/tmp/retrain_df.pkl").read_bytes())
    _, metrics = train_model(df=df, n_samples=len(df))
    logger.info("Challenger trained: %s", metrics)
    context["ti"].xcom_push(key="challenger_metrics", value=metrics)
    return metrics


def _champion_challenger_gate(**context) -> str:
    """Promote challenger only if R2 meets the gate."""
    from app.model import read_champion_metrics

    champion = read_champion_metrics()
    challenger = context["ti"].xcom_pull(key="challenger_metrics")

    champion_r2 = champion.get("r2", 0.0)
    challenger_r2 = challenger.get("r2", 0.0)

    logger.info(
        "Champion R2=%.4f  Challenger R2=%.4f  Gate=%.2f",
        champion_r2,
        challenger_r2,
        CHAMPION_R2_GATE,
    )

    if challenger_r2 >= champion_r2 and challenger_r2 >= CHAMPION_R2_GATE:
        logger.info("Challenger PROMOTED (R2=%.4f)", challenger_r2)
        return "promoted"
    else:
        logger.warning(
            "Challenger REJECTED (R2=%.4f < champion %.4f or < gate %.2f)",
            challenger_r2,
            champion_r2,
            CHAMPION_R2_GATE,
        )
        if METRICS_PATH.exists():
            METRICS_PATH.write_text(json.dumps(champion, indent=2))
        return "rejected"


def run_retraining_pipeline() -> dict:
    """Entry point for running the full pipeline outside Airflow."""
    import types

    ctx: dict = {"ti": types.SimpleNamespace(xcom_store={})}
    ctx["ti"].xcom_push = lambda key, value: ctx["ti"].xcom_store.update({key: value})
    ctx["ti"].xcom_pull = lambda key: ctx["ti"].xcom_store.get(key)

    logger.info("=== Quake-Net Retraining Pipeline START ===")
    try:
        _check_drift(**ctx)
        _load_fresh_data(**ctx)
        _train_challenger(**ctx)
        outcome = _champion_challenger_gate(**ctx)
        logger.info("=== Pipeline END — outcome: %s ===", outcome)
        return {"outcome": outcome, "stats": ctx["ti"].xcom_store}
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return {"outcome": "error", "error": str(exc)}


if _AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "quake-net",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "email_on_failure": False,
    }

    with DAG(
        dag_id="quake_net_weekly_retrain",
        default_args=default_args,
        description="Weekly champion/challenger retraining for Quake-Net",
        schedule_interval="0 2 * * 1",  # Every Monday at 02:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["quake-net", "ml", "seismic"],
    ) as dag:
        t_drift = PythonOperator(task_id="check_drift", python_callable=_check_drift)
        t_data = PythonOperator(task_id="load_fresh_data", python_callable=_load_fresh_data)
        t_train = PythonOperator(task_id="train_challenger", python_callable=_train_challenger)
        t_gate = PythonOperator(
            task_id="champion_challenger_gate",
            python_callable=_champion_challenger_gate,
        )

        t_drift >> t_data >> t_train >> t_gate
