"""Tests for Fleet-Mind model registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def temp_registry(tmp_path, monkeypatch):
    reg_path = tmp_path / "registry.json"
    import app.model_registry as mr
    mr._REGISTRY_PATH = reg_path
    yield reg_path
    mr._REGISTRY_PATH = Path("/tmp/fleet_mind_model_registry.json")


def test_register_model_creates_entry(temp_registry):
    import app.model_registry as mr
    mr.register_model("1.0.0", {"accuracy": 0.95}, "/tmp/model.joblib")
    entries = mr.list_models()
    assert len(entries) == 1
    assert entries[0]["version"] == "1.0.0"


def test_register_model_stores_metrics(temp_registry):
    import app.model_registry as mr
    metrics = {"roc_auc": 0.98, "f1": 0.91}
    mr.register_model("1.1.0", metrics, "/tmp/m.joblib")
    entries = mr.list_models()
    assert entries[0]["metrics"] == metrics


def test_register_multiple_models(temp_registry):
    import app.model_registry as mr
    mr.register_model("1.0.0", {}, "/tmp/m1.joblib")
    mr.register_model("1.1.0", {}, "/tmp/m2.joblib")
    entries = mr.list_models()
    assert len(entries) == 2


def test_get_latest_model_returns_last_registered(temp_registry):
    import app.model_registry as mr
    mr.register_model("1.0.0", {"acc": 0.9}, "/tmp/m1.joblib")
    mr.register_model("1.1.0", {"acc": 0.95}, "/tmp/m2.joblib")
    latest = mr.get_latest_model()
    assert latest is not None
    assert latest["version"] == "1.1.0"


def test_get_latest_model_empty_registry(temp_registry):
    import app.model_registry as mr
    latest = mr.get_latest_model()
    assert latest is None


def test_list_models_empty_registry(temp_registry):
    import app.model_registry as mr
    assert mr.list_models() == []


def test_registered_model_has_trained_at(temp_registry):
    import app.model_registry as mr
    mr.register_model("1.0.0", {}, "/tmp/m.joblib")
    entry = mr.list_models()[0]
    assert "trained_at" in entry


def test_registered_model_stores_path(temp_registry):
    import app.model_registry as mr
    mr.register_model("1.0.0", {}, "/models/best.joblib")
    entry = mr.list_models()[0]
    assert entry["model_path"] == "/models/best.joblib"
