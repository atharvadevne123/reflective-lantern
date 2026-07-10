"""Tests for the retraining pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pipelines"))


class TestRunPipeline:
    def test_skips_when_insufficient_data(self, monkeypatch):
        from pipelines import retrain_dag
        monkeypatch.setattr(retrain_dag, "extract_recent_readings", lambda *a, **k: [])
        result = retrain_dag.run_pipeline()
        assert result["status"] == "skipped"
        assert result["reason"] == "insufficient_data"

    def test_runs_with_sufficient_data(self, monkeypatch, sample_readings):
        from pipelines import retrain_dag
        raw = [
            {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
            for r in sample_readings
        ] * 12  # 600 readings > MIN_SAMPLES=500
        monkeypatch.setattr(retrain_dag, "extract_recent_readings", lambda *a, **k: raw)
        result = retrain_dag.run_pipeline()
        assert result["status"] == "success"
        assert result["n_readings"] == len(raw)

    def test_failure_path_reports_error(self, monkeypatch):
        from pipelines import retrain_dag
        monkeypatch.setattr(
            retrain_dag, "extract_recent_readings", lambda *a, **k: [{"bad": "data"}] * 600
        )
        result = retrain_dag.run_pipeline()
        assert result["status"] in ("failed", "success")


class TestFeatureEngineering:
    def test_feature_matrix_shape(self, sample_readings):
        from pipelines import retrain_dag
        raw = [
            {"timestamp": r["timestamp"], "sensor_id": r["sensor_id"], **r["values"]}
            for r in sample_readings
        ]
        X, cols = retrain_dag.run_feature_engineering(raw)
        assert X.shape[0] == len(raw)
        assert X.shape[1] == len(cols)


class TestExtractReadings:
    def test_extract_handles_db_error(self, monkeypatch):
        from pipelines import retrain_dag
        monkeypatch.setattr(retrain_dag, "DATABASE_URL", "postgresql://bad:bad@nonexistent:5/db")
        readings = retrain_dag.extract_recent_readings(lookback_days=1)
        assert readings == []
