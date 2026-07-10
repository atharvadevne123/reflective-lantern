"""MLflow-compatible experiment tracking stub for Forge-Guard.

Logs training metrics and model artefacts to the local filesystem in the
MLflow experiment format. Falls back gracefully when mlflow is not installed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MLFLOW_DIR = Path("mlruns/0")


def log_run(metrics: dict[str, Any], params: dict[str, Any] | None = None) -> str:
    """Write a minimal MLflow-compatible run record to mlruns/.

    Returns the run_id so callers can reference it in downstream artefacts.
    """
    try:
        import mlflow  # type: ignore[import]

        with mlflow.start_run():
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            if params:
                mlflow.log_params(params)
            run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
            logger.info("MLflow run logged: %s", run_id)
            return run_id
    except ImportError:
        pass

    # Fallback: write JSON manifest to mlruns/
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = MLFLOW_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run_id": run_id, "metrics": metrics, "params": params or {}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Fallback MLflow run written to %s", run_dir)
    return run_id
