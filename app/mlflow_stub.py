"""MLflow experiment tracking stub — logs metrics locally when MLflow is absent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_RUN_LOG = Path("mlflow_runs.jsonl")
_TRACKING_URI: str = ""


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


def log_training_run(
    params: dict[str, object],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    run_name: str = "training-run",
) -> str | None:
    """Log a training run with params, metrics, and optional tags.

    Returns None when no tracking URI is configured.

    Args:
        params: Hyperparameter key-value pairs.
        metrics: Metric key-value pairs.
        tags: Optional metadata tags.
        run_name: Human-readable run identifier.

    Returns:
        The run name on success, None if no tracking URI configured.
    """
    if not _TRACKING_URI:
        logger.debug("MLflow tracking URI not set; skipping training run log.")
        return None
    entry = {"run": run_name, "params": params, "metrics": metrics, "tags": tags or {}}
    with open(_RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("MLflow stub: logged training run '%s'", run_name)
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


def set_tracking_uri(uri: str) -> None:
    """Configure the MLflow tracking URI.

    Args:
        uri: Tracking server URI (e.g. 'http://localhost:5000').
    """
    global _TRACKING_URI
    _TRACKING_URI = uri
    logger.info("MLflow tracking URI set to '%s'", uri)


def log_params(run_name: str, params: dict[str, object]) -> None:
    """Append hyperparameter values to the local run log.

    Args:
        run_name: Identifier for the training run.
        params: Mapping of parameter name to value.
    """
    entry = {"run": run_name, "params": params}
    with open(_RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("MLflow stub: logged params for run '%s'", run_name)


def log_artifact(run_name: str, artifact_path: str) -> None:
    """Record an artifact path in the local run log.

    Args:
        run_name: Identifier for the training run.
        artifact_path: Local or remote path to the artefact file.
    """
    entry = {"run": run_name, "artifact": artifact_path}
    with open(_RUN_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    logger.info("MLflow stub: logged artifact '%s' for run '%s'", artifact_path, run_name)


def list_runs() -> list[dict[str, object]]:
    """Return all logged runs from the local JSONL log.

    Returns:
        List of run entry dicts, oldest first. Empty list when no log exists.
    """
    if not _RUN_LOG.exists():
        return []
    runs = []
    with open(_RUN_LOG) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed run log entry")
    return runs

__all__ = [
    "log_metrics",
    "log_training_run",
    "get_best_run",
    "set_tracking_uri",
    "log_params",
    "log_artifact",
    "list_runs",
]
