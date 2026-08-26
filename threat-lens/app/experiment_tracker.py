"""MLflow-compatible experiment tracking with a local fallback.

When MLflow is installed and `MLFLOW_TRACKING_URI` is set, runs are forwarded to
it. Otherwise runs are appended to a local JSONL file so training history is
still captured in environments without a tracking server.
"""

import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LOCAL_RUNS_PATH = Path(os.getenv("EXPERIMENT_LOG", "experiment_runs.jsonl"))
_HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


class ExperimentTracker:
    """Records training runs to MLflow when available, otherwise to disk."""

    def __init__(self, experiment_name: str = "threat-lens") -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.enabled = bool(_HAS_MLFLOW and self.tracking_uri)
        if not self.enabled:
            logger.info(
                "MLflow unavailable or MLFLOW_TRACKING_URI unset; logging runs to %s",
                LOCAL_RUNS_PATH,
            )

    def log_run(
        self,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> bool:
        """Record one training run.

        Args:
            params: Hyper-parameters used for the run.
            metrics: Resulting metric values.
            tags: Optional free-form tags.

        Returns:
            True if the run reached MLflow, False if it fell back to disk.
        """
        record = {
            "experiment": self.experiment_name,
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
        }

        if self.enabled:
            try:
                import mlflow  # noqa: PLC0415

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                with mlflow.start_run():
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    if tags:
                        mlflow.set_tags(tags)
                logger.info("Run logged to MLflow at %s", self.tracking_uri)
                return True
            except Exception:
                # Tracking must never take down a training job.
                logger.exception("MLflow logging failed; falling back to local file")

        self._append_local(record)
        return False

    @staticmethod
    def _append_local(record: dict[str, Any]) -> None:
        """Append one run record to the local JSONL log."""
        with LOCAL_RUNS_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def read_local_runs(path: Path | None = None) -> list[dict[str, Any]]:
        """Read every locally recorded run, skipping malformed lines."""
        target = path or LOCAL_RUNS_PATH
        if not target.exists():
            return []
        runs: list[dict[str, Any]] = []
        for line in target.read_text().splitlines():
            if not line.strip():
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed run record")
        return runs
