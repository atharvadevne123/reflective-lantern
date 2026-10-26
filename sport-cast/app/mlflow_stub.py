"""MLflow experiment tracking stub for Sport-Cast model logging."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_run(metrics: dict, params: dict | None = None, tags: dict | None = None) -> str:
    """Log a training run to MLflow (stub: logs locally when MLflow unavailable).

    Args:
        metrics: Dict of metric name -> float value.
        params: Optional dict of parameter name -> value.
        tags: Optional dict of tag name -> value.

    Returns:
        MLflow run_id string, or 'local-stub' if MLflow is not installed.
    """
    try:
        import mlflow

        with mlflow.start_run() as run:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if params:
                mlflow.log_params(params)
            if tags:
                mlflow.set_tags(tags)
            run_id = run.info.run_id
            logger.info("MLflow run logged: %s", run_id)
            return run_id
    except ImportError:
        logger.info("MLflow not installed — writing run locally: %s", metrics)
        _write_local_run(metrics, params or {}, tags or {})
        return "local-stub"
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): %s", exc)
        return "error-stub"


def _write_local_run(metrics: dict, params: dict, tags: dict) -> None:
    """Persist run information to a local JSON file as a fallback.

    Args:
        metrics: Metric values to persist.
        params: Parameter values to persist.
        tags: Tag values to persist.
    """
    run_path = Path("mlruns_local.jsonl")
    import time

    entry = {"ts": time.time(), "metrics": metrics, "params": params, "tags": tags}
    with run_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
