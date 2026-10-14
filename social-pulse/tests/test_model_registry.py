"""Tests for the model registry module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import app.model_registry as registry_module
from app.model_registry import (
    get_latest_promoted,
    list_models,
    promote_model,
    register_model,
)


@pytest.fixture(autouse=True)
def temp_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the registry module at a fresh temp file for each test."""
    reg_path = tmp_path / "registry.json"
    monkeypatch.setattr(registry_module, "REGISTRY_PATH", reg_path)
    yield reg_path


class TestRegisterModel:
    def test_creates_entry(self) -> None:
        entry = register_model("run-001", "1.0.0", 0.85, 0.72, 500)
        assert entry["run_id"] == "run-001"

    def test_entry_has_auc(self) -> None:
        entry = register_model("run-002", "1.0.0", 0.90, 0.80, 300)
        assert entry["auc_roc"] == pytest.approx(0.9, abs=1e-4)

    def test_entry_persisted(self, temp_registry: Path) -> None:
        register_model("run-003", "1.0.0", 0.75, 0.60, 200)
        data = json.loads(temp_registry.read_text())
        assert len(data) == 1

    def test_multiple_registrations(self) -> None:
        for i in range(3):
            register_model(f"run-{i:03d}", "1.0.0", 0.8, 0.7, 100)
        models = list_models()
        assert len(models) == 3


class TestListModels:
    def test_empty_registry(self) -> None:
        assert list_models() == []

    def test_newest_first(self) -> None:
        register_model("first", "1.0.0", 0.7, 0.6, 100)
        register_model("second", "1.0.1", 0.8, 0.7, 200)
        models = list_models()
        assert models[0]["run_id"] == "second"


class TestGetLatestPromoted:
    def test_none_when_no_promoted(self) -> None:
        register_model("run-x", "1.0.0", 0.8, 0.7, 100)
        assert get_latest_promoted() is None

    def test_returns_promoted_model(self) -> None:
        register_model("run-a", "1.0.0", 0.8, 0.7, 100, promoted=True)
        promoted = get_latest_promoted()
        assert promoted is not None
        assert promoted["run_id"] == "run-a"


class TestPromoteModel:
    def test_promote_existing(self) -> None:
        register_model("run-p1", "1.0.0", 0.8, 0.7, 100)
        result = promote_model("run-p1")
        assert result is True

    def test_promote_nonexistent_returns_false(self) -> None:
        result = promote_model("does-not-exist")
        assert result is False

    def test_only_one_promoted_at_a_time(self) -> None:
        register_model("r1", "1.0.0", 0.8, 0.7, 100, promoted=True)
        register_model("r2", "1.0.1", 0.85, 0.75, 200)
        promote_model("r2")
        models = list_models()
        promoted = [m for m in models if m["promoted"]]
        assert len(promoted) == 1
        assert promoted[0]["run_id"] == "r2"


@pytest.mark.parametrize("auc,f1", [(0.5, 0.4), (0.8, 0.7), (1.0, 1.0)])
def test_register_various_metrics(auc: float, f1: float) -> None:
    entry = register_model("run-v", "1.0.0", auc, f1, 100)
    assert entry["auc_roc"] == pytest.approx(auc, abs=1e-4)
    assert entry["f1"] == pytest.approx(f1, abs=1e-4)
