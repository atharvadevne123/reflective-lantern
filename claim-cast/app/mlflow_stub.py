"""MLflow experiment tracking stub — swappable for real MLflow in production."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "claim-cast"
_runs: list[dict] = []


def start_run(run_name: str = "train") -> dict:
    """Begin a new MLflow-compatible experiment run.

    Args:
        run_name: Human-readable name for this run.

    Returns:
        Run metadata dict with run_id and start_time.
    """
    run = {
        "run_id": f"stub_{len(_runs):04d}",
        "run_name": run_name,
        "start_time": datetime.utcnow().isoformat(),
        "params": {},
        "metrics": {},
        "status": "RUNNING",
    }
    _runs.append(run)
    return run


def log_param(run: dict, key: str, value: object) -> None:
    """Log a hyperparameter to the run."""
    run["params"][key] = str(value)


def log_metric(run: dict, key: str, value: float) -> None:
    """Log a numeric metric to the run."""
    run["metrics"][key] = value


def end_run(run: dict) -> None:
    """Mark the run as finished and persist to local JSONL log."""
    run["status"] = "FINISHED"
    run["end_time"] = datetime.utcnow().isoformat()
    log_path = Path("mlflow_runs.jsonl")
    with log_path.open("a") as f:
        f.write(json.dumps(run) + "\n")


def get_runs() -> list[dict]:
    """Return all in-memory run records."""
    return list(_runs)
