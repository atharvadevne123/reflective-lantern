"""Tests for threat-lens app/experiment_tracker.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    def test_init_default_name(self) -> None:
        tracker = ExperimentTracker()
        assert tracker.experiment_name == "threat-lens"

    def test_init_custom_name(self) -> None:
        tracker = ExperimentTracker("my-experiment")
        assert tracker.experiment_name == "my-experiment"

    def test_disabled_without_mlflow_uri(self) -> None:
        with patch("app.experiment_tracker._HAS_MLFLOW", False):
            tracker = ExperimentTracker()
            assert tracker.enabled is False

    def test_log_run_returns_bool(self) -> None:
        tracker = ExperimentTracker()
        result = tracker.log_run(
            params={"n_estimators": 100},
            metrics={"accuracy": 0.95},
        )
        assert isinstance(result, bool)

    def test_log_run_without_mlflow(self) -> None:
        with patch("app.experiment_tracker._HAS_MLFLOW", False):
            tracker = ExperimentTracker()
            result = tracker.log_run(params={}, metrics={"rmse": 0.1})
        assert isinstance(result, bool)

    def test_log_run_with_tags(self) -> None:
        tracker = ExperimentTracker()
        result = tracker.log_run(
            params={"lr": 0.01},
            metrics={"f1": 0.88},
            tags={"env": "test"},
        )
        assert isinstance(result, bool)

    @pytest.mark.parametrize(
        "params,metrics",
        [
            ({"n_estimators": 50}, {"accuracy": 0.9}),
            ({"max_depth": 5}, {"f1": 0.85}),
            ({}, {"auc": 0.92}),
        ],
    )
    def test_various_metrics(self, params: dict, metrics: dict) -> None:
        tracker = ExperimentTracker()
        result = tracker.log_run(params=params, metrics=metrics)
        assert isinstance(result, bool)
