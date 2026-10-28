"""Automated retraining pipeline — Airflow-compatible DAG definition."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "fleet-mind",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_retraining() -> dict:
    """Execute model retraining and return metrics."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.model import train_model
    metrics = train_model()
    logger.info("Retraining complete: %s", metrics)
    return metrics


def check_drift() -> list[dict]:
    """Run drift detection and return flagged features."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.database import _get_session_factory, init_db
    from app.monitoring import run_drift_check
    init_db()
    factory = _get_session_factory()
    db = factory()
    try:
        reports = run_drift_check(db)
        drifted = [r for r in reports if r.get("drift_detected")]
        if drifted:
            logger.warning("Drift in %d feature(s): %s", len(drifted), [r["feature"] for r in drifted])
        return drifted
    finally:
        db.close()


def save_metrics_artifact(metrics: dict, output_dir: str = "/tmp/fleet_mind_artifacts") -> str:
    """Persist metrics as a JSON artifact."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = f"{output_dir}/metrics_{ts}.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics artifact saved: %s", path)
    return path


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="fleet_mind_retrain",
        default_args=DEFAULT_ARGS,
        description="Periodic Fleet-Mind model retraining with drift check",
        schedule="0 2 * * *",  # 02:00 UTC daily
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["fleet-mind", "ml", "retraining"],
    ) as dag:
        drift_task = PythonOperator(
            task_id="check_drift",
            python_callable=check_drift,
        )

        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=run_retraining,
        )

        save_task = PythonOperator(
            task_id="save_metrics",
            python_callable=lambda **ctx: save_metrics_artifact(
                ctx["ti"].xcom_pull(task_ids="retrain_model")
            ),
        )

        drift_task >> retrain_task >> save_task  # type: ignore[operator]

except ImportError:
    # Airflow not installed — DAG definition is skipped silently
    pass


if __name__ == "__main__":
    drifted = check_drift()
    print(f"Drifted features: {drifted}")
    metrics = run_retraining()
    print(f"Retraining metrics: {metrics}")
    artifact = save_metrics_artifact(metrics)
    print(f"Artifact: {artifact}")
