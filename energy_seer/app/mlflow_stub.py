"""MLflow experiment tracking stub — swappable for a real MLflow client."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_params(params: dict) -> None:
    """Log hyperparameters to MLflow (or just locally)."""
    logger.info("mlflow_params %s", params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log evaluation metrics to MLflow."""
    logger.info("mlflow_metrics step=%s %s", step, metrics)


def log_model(model_path: str, artifact_name: str = "model") -> None:
    """Log a model artifact to MLflow."""
    logger.info("mlflow_artifact name=%s path=%s", artifact_name, model_path)


def start_run(run_name: str = "energy-seer") -> None:
    logger.info("mlflow_run_start name=%s", run_name)


def end_run() -> None:
    logger.info("mlflow_run_end")
