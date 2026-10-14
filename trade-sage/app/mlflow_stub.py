"""MLflow integration stub for experiment tracking.

In production, set MLFLOW_TRACKING_URI to enable real logging.
Without it, this module is a no-op that lets the rest of the codebase
call MLflow-style APIs without failing on import.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "")

try:
    if MLFLOW_URI:
        import mlflow  # noqa: F401
        _MLFLOW_AVAILABLE = True
    else:
        _MLFLOW_AVAILABLE = False
except ImportError:
    _MLFLOW_AVAILABLE = False


def log_run(run_name: str, metrics: dict, params: dict | None = None) -> None:
    """Log a training run to MLflow if available.

    Args:
        run_name: Human-readable run identifier.
        metrics: Dict of metric name → float value.
        params: Optional dict of hyperparameter name → value.
    """
    if not _MLFLOW_AVAILABLE:
        logger.debug("MLflow not configured — skipping run log for %s", run_name)
        return
    import mlflow

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    logger.info("MLflow run logged: %s", run_name)


def log_model(model, artifact_path: str = "model") -> None:
    """Log a sklearn-compatible model to MLflow if available.

    Args:
        model: Fitted sklearn Pipeline or estimator.
        artifact_path: MLflow artifact subdirectory.
    """
    if not _MLFLOW_AVAILABLE:
        return
    import mlflow.sklearn

    mlflow.sklearn.log_model(model, artifact_path)
