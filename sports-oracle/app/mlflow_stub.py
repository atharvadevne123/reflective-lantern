"""MLflow experiment tracking stub for Sports-Oracle."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MLFLOW_RUNS_PATH = Path("mlflow_runs.jsonl")


def log_run(
    run_name: str,
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training run (stub: writes to JSONL file instead of MLflow server)."""
    import uuid

    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id": run_id,
        "run_name": run_name,
        "metrics": metrics,
        "params": params or {},
        "tags": tags or {},
    }
    try:
        with MLFLOW_RUNS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.warning("mlflow_stub_write_failed", extra={"run_id": run_id})
    logger.info("mlflow_run_logged", extra={"run_id": run_id, "metrics": metrics})
    return run_id


def get_best_run(metric: str = "auc_mean") -> dict | None:
    """Return the run with the highest value for the given metric."""
    if not MLFLOW_RUNS_PATH.exists():
        return None
    best: dict | None = None
    best_val = float("-inf")
    try:
        for line in MLFLOW_RUNS_PATH.read_text().splitlines():
            record = json.loads(line)
            val = record.get("metrics", {}).get(metric, float("-inf"))
            if val > best_val:
                best_val = val
                best = record
    except Exception:
        logger.exception("mlflow_stub_read_failed")
    return best
