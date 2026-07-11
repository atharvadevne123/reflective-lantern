"""MLflow experiment tracking stub — logs metrics locally when MLflow is absent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_RUN_LOG = Path("mlflow_runs.jsonl")


def log_metrics(run_name: str, metrics: dict[str, float]) -> str:
    """Append training metrics to the local JSONL run log.

    Acts as a drop-in stub for ``mlflow.log_metrics`` when MLflow is not
    configured.  Each call appends one JSON line to ``mlflow_runs.jsonl``.

    Args:
        run_name: Human-readable identifier for this training run.
        metrics: Mapping of metric name to float value (e.g. ``{"r2": 0.91}``).

    Returns:
        The same *run_name* string that was passed in.
    """
    entry = {"run": run_name, "metrics": metrics}
    with open(_RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("MLflow stub: logged run '%s'  metrics=%s", run_name, metrics)
    return run_name


def get_best_run(metric: str = "r2_mean") -> dict[str, object] | None:
    """Return the logged run that achieved the highest value for *metric*.

    Args:
        metric: Name of the metric to rank by (default ``"r2_mean"``).

    Returns:
        The best run entry dict, or ``None`` if the run log does not exist.
    """
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
