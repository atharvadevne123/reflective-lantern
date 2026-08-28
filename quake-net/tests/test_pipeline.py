"""Tests for the automated champion/challenger retraining pipeline."""

from __future__ import annotations

import json
import types

import pytest

from pipelines import retrain_dag
from pipelines.retrain_dag import (
    CHAMPION_R2_GATE,
    MIN_SAMPLES,
    _champion_challenger_gate,
    _load_fresh_data,
    run_retraining_pipeline,
)


def _context() -> dict:
    store: dict = {}
    ti = types.SimpleNamespace(xcom_store=store)
    ti.xcom_push = lambda key, value: store.update({key: value})
    ti.xcom_pull = lambda key: store.get(key)
    return {"ti": ti}


class TestLoadFreshData:
    def test_returns_sample_statistics(self) -> None:
        stats = _load_fresh_data(**_context())
        assert stats["n_samples"] >= MIN_SAMPLES
        assert "magnitude_mean" in stats

    def test_pushes_stats_to_xcom(self) -> None:
        ctx = _context()
        _load_fresh_data(**ctx)
        assert ctx["ti"].xcom_pull("data_stats") is not None


class TestChampionChallengerGate:
    def test_promotes_stronger_challenger(self, tmp_path, monkeypatch) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"r2": 0.75}))
        monkeypatch.setattr(retrain_dag, "METRICS_PATH", metrics_file)
        monkeypatch.setattr("app.model.METRICS_PATH", metrics_file, raising=False)

        ctx = _context()
        ctx["ti"].xcom_push("challenger_metrics", {"r2": 0.88})
        assert _champion_challenger_gate(**ctx) == "promoted"

    def test_rejects_weaker_challenger(self, tmp_path, monkeypatch) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"r2": 0.90}))
        monkeypatch.setattr(retrain_dag, "METRICS_PATH", metrics_file)
        monkeypatch.setattr("app.model.METRICS_PATH", metrics_file, raising=False)

        ctx = _context()
        ctx["ti"].xcom_push("challenger_metrics", {"r2": 0.55})
        assert _champion_challenger_gate(**ctx) == "rejected"

    def test_rejects_challenger_below_absolute_gate(self, tmp_path, monkeypatch) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"r2": 0.10}))
        monkeypatch.setattr(retrain_dag, "METRICS_PATH", metrics_file)
        monkeypatch.setattr("app.model.METRICS_PATH", metrics_file, raising=False)

        ctx = _context()
        # Beats the weak champion but still misses the absolute quality gate.
        ctx["ti"].xcom_push("challenger_metrics", {"r2": CHAMPION_R2_GATE - 0.05})
        assert _champion_challenger_gate(**ctx) == "rejected"

    def test_rejection_restores_champion_metrics(self, tmp_path, monkeypatch) -> None:
        metrics_file = tmp_path / "metrics.json"
        champion = {"r2": 0.92, "rmse": 0.31}
        metrics_file.write_text(json.dumps(champion))
        monkeypatch.setattr(retrain_dag, "METRICS_PATH", metrics_file)
        monkeypatch.setattr("app.model.METRICS_PATH", metrics_file, raising=False)

        ctx = _context()
        ctx["ti"].xcom_push("challenger_metrics", {"r2": 0.40})
        _champion_challenger_gate(**ctx)
        assert json.loads(metrics_file.read_text())["r2"] == 0.92

    def test_missing_champion_treated_as_zero(self, tmp_path, monkeypatch) -> None:
        metrics_file = tmp_path / "absent.json"
        monkeypatch.setattr(retrain_dag, "METRICS_PATH", metrics_file)
        monkeypatch.setattr("app.model.METRICS_PATH", metrics_file, raising=False)

        ctx = _context()
        ctx["ti"].xcom_push("challenger_metrics", {"r2": 0.85})
        assert _champion_challenger_gate(**ctx) == "promoted"


class TestGateConstants:
    def test_gate_is_a_sane_threshold(self) -> None:
        assert 0.0 < CHAMPION_R2_GATE < 1.0

    def test_min_samples_positive(self) -> None:
        assert MIN_SAMPLES > 0


class TestRunRetrainingPipeline:
    @pytest.mark.slow
    def test_pipeline_reports_an_outcome(self) -> None:
        result = run_retraining_pipeline()
        assert result["outcome"] in {"promoted", "rejected", "error"}
