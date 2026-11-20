"""MLflow experiment tracking, degrading gracefully when MLflow is absent.

Tracking is an operational nicety, not a hard dependency: a training run must
still succeed on a machine without MLflow installed or without a reachable
tracking server. Every entry point here is therefore best-effort and swallows
backend errors after logging them.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "cyber-guard")

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without mlflow
    _MLFLOW_AVAILABLE = False


def is_tracking_enabled() -> bool:
    """Return True when MLflow is importable and a tracking URI is configured."""
    return _MLFLOW_AVAILABLE and bool(MLFLOW_TRACKING_URI)


@contextmanager
def track_run(run_name: str):
    """Open an MLflow run, or a no-op context when tracking is unavailable.

    Args:
        run_name: Human-readable name for the run.

    Yields:
        None. The context is a no-op unless tracking is enabled.
    """
    if not is_tracking_enabled():
        logger.debug("mlflow tracking disabled, skipping run %s", run_name)
        yield
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            yield
    except Exception as exc:  # pragma: no cover - depends on a live server
        # A tracking outage must never fail a training run.
        logger.warning("mlflow run failed, continuing untracked: %s", exc)
        yield


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log numeric metrics to the active MLflow run, if any.

    Non-numeric values are skipped rather than raising.

    Args:
        metrics: Mapping of metric name to value.
    """
    if not is_tracking_enabled():
        return
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    try:
        mlflow.log_metrics(numeric)
    except Exception as exc:  # pragma: no cover - depends on a live server
        logger.warning("mlflow metric logging failed: %s", exc)


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the active MLflow run, if any.

    Args:
        params: Mapping of parameter name to value.
    """
    if not is_tracking_enabled():
        return
    try:
        mlflow.log_params(params)
    except Exception as exc:  # pragma: no cover - depends on a live server
        logger.warning("mlflow param logging failed: %s", exc)
