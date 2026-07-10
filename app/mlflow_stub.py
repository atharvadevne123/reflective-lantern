"""MLflow stub for experiment tracking — swappable with real MLflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MLFLOW_ENABLED = False
_RUN_STORE: list[dict] = []


def log_metrics(metrics: dict, run_name: str = "volt-cast") -> str:
    """Log model metrics — writes to local store or MLflow if enabled."""
    run_id = f"run_{len(_RUN_STORE):04d}"
    entry = {"run_id": run_id, "run_name": run_name, **metrics}
    _RUN_STORE.append(entry)

    if MLFLOW_ENABLED:
        try:
            import mlflow  # type: ignore[import]
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as exc:
            logger.warning("mlflow logging failed: %s", exc)
    else:
        logger.info("mlflow stub — logged run %s: %s", run_id, metrics)

    return run_id


def get_best_run(metric: str = "r2_mean") -> dict | None:
    """Return the run with the highest value for `metric`."""
    valid = [r for r in _RUN_STORE if metric in r]
    if not valid:
        return None
    return max(valid, key=lambda r: r[metric])


def dump_run_store(path: str = "/tmp/mlflow_stub_runs.json") -> None:
    Path(path).write_text(json.dumps(_RUN_STORE, indent=2))
    logger.info("mlflow stub runs written to %s", path)
