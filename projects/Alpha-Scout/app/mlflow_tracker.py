"""MLflow experiment tracking stub."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_ENABLED = True
except ImportError:
    _MLFLOW_ENABLED = False
    logger.info("MLflow not installed; tracking disabled")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "alpha-scout")


def log_training_run(metrics: dict[str, Any], params: dict[str, Any] | None = None) -> str:
    """Log a training run to MLflow if available; returns run_id or empty string."""
    if not _MLFLOW_ENABLED:
        logger.info("MLflow stub: metrics=%s params=%s", metrics, params)
        return ""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        return run.info.run_id


def log_prediction_event(ticker: str, alpha_score: float, confidence: float) -> None:
    """Log a single prediction event as an MLflow metric (optional)."""
    if not _MLFLOW_ENABLED:
        return
    try:
        with mlflow.start_run(run_name=f"predict_{ticker}", nested=True):
            mlflow.log_metrics({"alpha_score": alpha_score, "confidence": confidence})
    except Exception as exc:
        logger.warning("MLflow prediction log failed: %s", exc)
