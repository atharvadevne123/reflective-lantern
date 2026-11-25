"""MLflow experiment tracking for model training runs."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def log_training_run(metrics: dict[str, Any], params: dict[str, Any] | None = None) -> None:
    """Log a training run to MLflow if available."""
    if not _MLFLOW_AVAILABLE:
        logger.info("MLflow not available - metrics: %s", metrics)
        return
    with mlflow.start_run():
        for k, v in (params or {}).items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        logger.info("MLflow run logged: %s", metrics)
