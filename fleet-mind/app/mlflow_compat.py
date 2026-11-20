"""MLflow-compatible experiment tracking stub.

When mlflow is installed, logs runs to the configured tracking URI.
When it is absent, falls back to a local JSONL file so the rest of the
codebase can always call these functions unconditionally.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FALLBACK_LOG = Path(os.getenv("MLFLOW_FALLBACK_LOG", "/tmp/fleet_mind_mlflow.jsonl"))

try:
    import mlflow  # type: ignore[import]
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def log_training_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training run with params and metrics.

    Args:
        run_name: Human-readable name for the run.
        params: Hyperparameters or configuration values to log.
        metrics: Evaluation metrics to record.
        tags: Optional string tags to attach to the run.

    Returns:
        Run identifier string (MLflow run_id or ISO timestamp).
    """
    if _HAS_MLFLOW:
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            logger.info("MLflow run %s logged: %s", run.info.run_id, metrics)
            return run.info.run_id
    else:
        entry = {
            "run_name": run_name,
            "logged_at": datetime.utcnow().isoformat(),
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
        }
        try:
            with open(_FALLBACK_LOG, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("MLflow fallback log write failed: %s", exc)
        run_id = entry["logged_at"]
        logger.info("MLflow stub run %s logged: %s", run_id, metrics)
        return run_id
