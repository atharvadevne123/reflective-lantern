"""Tests for the model registry."""

import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.model_registry import ModelRegistry, ModelRecord


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(registry_path=tmp_path / "registry.json")


def test_register_returns_record(registry):
    rec = registry.register("1.0.0", "/tmp/model.pkl")
    assert isinstance(rec, ModelRecord)
    assert rec.version == "1.0.0"


def test_get_active_after_register(registry):
    registry.register("1.0.0", "/tmp/model.pkl", set_active=True)
    active = registry.get_active()
    assert active is not None
    assert active.version == "1.0.0"


def test_register_new_version_deactivates_old(registry):
    registry.register("1.0.0", "/tmp/m1.pkl", set_active=True)
    registry.register("1.1.0", "/tmp/m2.pkl", set_active=True)
    active = registry.get_active()
    assert active.version == "1.1.0"


def test_list_all_returns_newest_first(registry):
    registry.register("1.0.0", "/tmp/m1.pkl")
    registry.register("1.1.0", "/tmp/m2.pkl")
    records = registry.list_all()
    assert records[0].version == "1.1.0"


def test_empty_registry_get_active(registry):
    assert registry.get_active() is None


@pytest.mark.parametrize("version", ["1.0.0", "2.0.0-beta", "1.2.3"])
def test_register_various_versions(registry, version):
    rec = registry.register(version, "/tmp/model.pkl")
    assert rec.version == version
