"""Automated retraining pipeline for Property-Sage models."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from app.features import generate_synthetic_data
from app.model import train_model

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD_DRIFT = float(os.getenv("RETRAIN_DRIFT_THRESHOLD", "0.05"))
RETRAIN_MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "500"))
RETRAIN_LOG = Path("models/retrain_log.json")


def should_retrain(drift_results: list[dict]) -> bool:
    """Return True if any monitored feature shows statistically significant drift."""
    return any(d.get("drift_detected") and d.get("p_value", 1.0) < RETRAIN_THRESHOLD_DRIFT for d in drift_results)


def run_retrain_pipeline(
    n_samples: int = 2000,
    drift_results: list[dict] | None = None,
) -> dict:
    """Full retraining pipeline: generate data, train, evaluate, log."""
    started_at = datetime.utcnow().isoformat()
    logger.info("Retraining pipeline started at %s", started_at)

    if drift_results and not should_retrain(drift_results):
        logger.info("No significant drift detected — skipping retrain")
        return {"status": "skipped", "reason": "no_drift", "started_at": started_at}

    X, y_price, y_rental = generate_synthetic_data(n=n_samples)
    metrics = train_model(X, y_price, y_rental)

    finished_at = datetime.utcnow().isoformat()
    record = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "n_samples": n_samples,
        "metrics": metrics,
        "triggered_by_drift": bool(drift_results),
    }

    _append_log(record)
    logger.info("Retraining complete — price_r2=%.4f rental_r2=%.4f", metrics["price_r2_mean"], metrics["rental_r2_mean"])
    return record


def _append_log(record: dict) -> None:
    RETRAIN_LOG.parent.mkdir(exist_ok=True)
    history = []
    if RETRAIN_LOG.exists():
        try:
            history = json.loads(RETRAIN_LOG.read_text())
        except json.JSONDecodeError:
            history = []
    history.append(record)
    RETRAIN_LOG.write_text(json.dumps(history[-50:], indent=2))


def get_retrain_history() -> list[dict]:
    if RETRAIN_LOG.exists():
        try:
            return json.loads(RETRAIN_LOG.read_text())
        except json.JSONDecodeError:
            return []
    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_retrain_pipeline(n_samples=2000)
    print(json.dumps(result, indent=2))
