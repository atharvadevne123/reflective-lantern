"""MLflow experiment tracking stub — logs metrics locally when MLflow is absent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_RUN_LOG = Path("mlflow_runs.jsonl")


def log_metrics(run_name: str, metrics: dict[str, float]) -> str:
    """Log training metrics to a local JSONL file (stub for MLflow)."""
    entry = {"run": run_name, "metrics": metrics}
    with open(_RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("MLflow stub: logged run '%s'  metrics=%s", run_name, metrics)
    return run_name


def get_best_run(metric: str = "r2_mean") -> dict | None:
    """Return the run with the highest value for *metric*."""
    if not _RUN_LOG.exists():
        return None
    best = None
    best_val = float("-inf")
    with open(_RUN_LOG) as fh:
        for line in fh:
            entry = json.loads(line)
            val = entry["metrics"].get(metric, float("-inf"))
            if val > best_val:
                best_val = val
                best = entry
    return best
