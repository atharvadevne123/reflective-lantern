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


def is_mlflow_available() -> bool:
    """Return True when MLflow tracking is configured and importable."""
    return _MLFLOW_AVAILABLE


def log_run_safe(run_name: str, metrics: dict, params: dict | None = None) -> bool:
    """Log a training run; return True on success, False on any error or no-op.

    Unlike ``log_run``, this wrapper never raises — useful in background jobs
    where MLflow failures should be observed but not fatal.

    Args:
        run_name: Human-readable run identifier.
        metrics: Dict of metric name → float value.
        params: Optional dict of hyperparameter name → value.

    Returns:
        True when the run was logged, False otherwise.
    """
    if not _MLFLOW_AVAILABLE:
        return False
    try:
        log_run(run_name, metrics, params)
        return True
    except Exception as exc:
        logger.warning("MLflow log_run_safe failed: %s", exc)
        return False
