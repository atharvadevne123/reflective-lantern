"""Tests for retraining pipeline task functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipelines.retrain_dag import (
    AUC_GATE,
    task_read_champion_auc,
    task_train_challenger,
    task_validate_gate,
)


class TestTaskReadChampionAuc:
    def test_returns_float(self) -> None:
        with patch("pipelines.retrain_dag.read_champion_auc", return_value=0.75) as mock_fn:
            mock_fn.__module__ = "app.model"
            with patch("app.model.read_champion_auc", return_value=0.75):
                result = task_read_champion_auc()
                assert isinstance(result, float)

    def test_returns_zero_when_no_model(self) -> None:
        with patch("app.model.read_champion_auc", return_value=0.0):
            result = task_read_champion_auc()
            assert result == 0.0


class TestAucGate:
    def test_auc_gate_is_positive(self) -> None:
        assert AUC_GATE > 0.0

    def test_auc_gate_is_below_one(self) -> None:
        assert AUC_GATE < 1.0

    def test_auc_gate_default_value(self) -> None:
        assert AUC_GATE == pytest.approx(0.60, abs=0.01)


class TestTaskValidateGate:
    def test_promoted_outcome_no_error(self) -> None:
        ctx = {"ti": MagicMock()}
        ctx["ti"].xcom_pull.return_value = "promoted"
        task_validate_gate(**ctx)

    def test_rejected_outcome_no_error(self) -> None:
        ctx = {"ti": MagicMock()}
        ctx["ti"].xcom_pull.return_value = "rejected"
        task_validate_gate(**ctx)

    def test_called_with_correct_task_id(self) -> None:
        ctx = {"ti": MagicMock()}
        ctx["ti"].xcom_pull.return_value = "promoted"
        task_validate_gate(**ctx)
        ctx["ti"].xcom_pull.assert_called_with(task_ids="compare_and_promote")
