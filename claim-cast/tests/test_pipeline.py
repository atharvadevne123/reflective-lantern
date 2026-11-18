"""Tests for the automated retraining pipeline."""
from __future__ import annotations

import pytest


def test_fetch_training_data_creates_parquet(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    from pipelines.retrain_dag import fetch_training_data

    output = fetch_training_data()
    import os

    assert os.path.exists(output)


def test_run_retraining_produces_metrics(tmp_path):
    from pipelines.retrain_dag import fetch_training_data, run_retraining

    data_path = fetch_training_data()
    metrics = run_retraining(data_path)
    assert "f1_macro_mean" in metrics
    assert 0.0 <= metrics["f1_macro_mean"] <= 1.0


def test_evaluate_model_passes_above_threshold():
    from pipelines.retrain_dag import evaluate_model

    assert evaluate_model({"f1_macro_mean": 0.7}) is True


def test_evaluate_model_fails_below_threshold():
    from pipelines.retrain_dag import evaluate_model

    assert evaluate_model({"f1_macro_mean": 0.3}) is False


def test_evaluate_model_no_metrics():
    from pipelines.retrain_dag import evaluate_model

    result = evaluate_model({})
    assert isinstance(result, bool)
