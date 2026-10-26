"""MLflow experiment tracking stub for Ticket-Scout."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow  # type: ignore[import]
    _MLFLOW_AVAILABLE = True
except ImportError:
    pass

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "ticket-scout")


def log_training_run(metrics: dict, params: dict | None = None) -> str | None:
    """Log a training run to MLflow if available, else write to local JSON.

    Args:
        metrics: Model CV metrics to log.
        params: Hyperparameters to log.

    Returns:
        MLflow run ID if available, else None.
    """
    if _MLFLOW_AVAILABLE and TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            logger.info("MLflow run logged: %s", run.info.run_id)
            return run.info.run_id

    # Fallback: write to local file
    log_path = Path("models/mlflow_runs.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    entry = {"metrics": metrics, "params": params or {}}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Training run logged locally (MLflow not configured)")
    return None
