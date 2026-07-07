"""MLflow integration stub for Realty-Edge experiment tracking.

Logs model metrics, parameters, and artefacts to MLflow when the
MLFLOW_TRACKING_URI environment variable is set. Falls back to a no-op
logger when MLflow is unavailable (e.g. in CI or lightweight dev setups).

Usage
-----
from app.mlflow_stub import log_training_run
log_training_run(params={"n_estimators": 200}, metrics={"r2": 0.87}, tags={"env": "prod"})
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "realty-edge")


def log_training_run(
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifact_paths: list[str] | None = None,
) -> str | None:
    """Log a training run to MLflow if available.

    Args:
        params: Model hyperparameters to record.
        metrics: Evaluation metrics (R2, RMSE, etc.).
        tags: Arbitrary key-value metadata for the run.
        artifact_paths: Local file paths to upload as artefacts.

    Returns:
        The MLflow run ID if logging succeeded, else None.
    """
    if not _TRACKING_URI:
        logger.debug("MLFLOW_TRACKING_URI not set — skipping MLflow logging.")
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment(_EXPERIMENT)
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            for path in (artifact_paths or []):
                if os.path.exists(path):
                    mlflow.log_artifact(path)
            run_id = run.info.run_id
            logger.info("MLflow run logged: %s", run_id)
            return run_id
    except ImportError:
        logger.debug("MLflow not installed — skipping.")
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
    return None
