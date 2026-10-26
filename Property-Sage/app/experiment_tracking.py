"""Lightweight experiment tracking for Property-Sage model runs.

Provides a simple file-based experiment log that mirrors MLflow's concept
of runs and metrics without requiring an MLflow server. Each training run
is persisted as a JSON record, enabling reproducibility comparisons.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPERIMENT_DIR: Path = Path(os.getenv("EXPERIMENT_DIR", "experiments"))


def log_run(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training run with parameters, metrics, and optional tags.

    Args:
        experiment_name: Human-readable name for the experiment group.
        params: Hyperparameters and configuration used in this run.
        metrics: Computed evaluation metrics (R², RMSE, MAE, etc.).
        tags: Optional string metadata (model version, data source, etc.).

    Returns:
        Unique run_id string for this experiment record.
    """
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    record: dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "timestamp": datetime.utcnow().isoformat(),
        "params": params,
        "metrics": metrics,
        "tags": tags or {},
    }

    log_path = EXPERIMENT_DIR / f"{experiment_name}.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Experiment run logged — experiment=%s run_id=%s",
        experiment_name,
        run_id,
    )
    return run_id


def get_best_run(
    experiment_name: str,
    metric: str,
    higher_is_better: bool = True,
) -> dict[str, Any] | None:
    """Return the best run for an experiment according to a named metric.

    Args:
        experiment_name: Name of the experiment log file to search.
        metric: Key in the metrics dict to optimise.
        higher_is_better: If True, return the run with the highest metric value.

    Returns:
        The best run record dict, or None if no runs exist.
    """
    log_path = EXPERIMENT_DIR / f"{experiment_name}.jsonl"
    if not log_path.exists():
        return None

    runs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not runs:
        return None

    valid = [r for r in runs if metric in r.get("metrics", {})]
    if not valid:
        return None

    return (
        max(valid, key=lambda r: r["metrics"][metric])
        if higher_is_better
        else min(valid, key=lambda r: r["metrics"][metric])
    )


def list_runs(experiment_name: str) -> list[dict[str, Any]]:
    """List all runs for a given experiment.

    Args:
        experiment_name: Experiment name to query.

    Returns:
        List of run record dicts, newest first.
    """
    log_path = EXPERIMENT_DIR / f"{experiment_name}.jsonl"
    if not log_path.exists():
        return []
    runs = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return list(reversed(runs))
