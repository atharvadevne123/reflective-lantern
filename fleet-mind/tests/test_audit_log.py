"""Tests for Fleet-Mind audit log utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def temp_audit_log(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))
    # Force reload so the module picks up the new env var
    import app.audit_log as audit_module
    audit_module._AUDIT_LOG_PATH = log_path
    yield log_path


def test_log_prediction_request_creates_file(temp_audit_log):
    import app.audit_log as audit_module
    audit_module.log_prediction_request("VH-001", "127.0.0.1", "corr-123", True)
    assert temp_audit_log.exists()


def test_log_prediction_request_entry_has_required_fields(temp_audit_log):
    import app.audit_log as audit_module
    audit_module.log_prediction_request("VH-001", "10.0.0.1", "corr-abc", False)
    lines = temp_audit_log.read_text().strip().split("\n")
    entry = json.loads(lines[-1])
    assert entry["event"] == "prediction_request"
    assert entry["vehicle_id"] == "VH-001"
    assert entry["client_ip"] == "10.0.0.1"
    assert entry["correlation_id"] == "corr-abc"
    assert entry["maintenance_needed"] is False


def test_log_retrain_event_entry_has_required_fields(temp_audit_log):
    import app.audit_log as audit_module
    metrics = {"accuracy": 0.95, "roc_auc": 0.98}
    audit_module.log_retrain_event("admin", metrics, "corr-xyz")
    lines = temp_audit_log.read_text().strip().split("\n")
    entry = json.loads(lines[-1])
    assert entry["event"] == "model_retrain"
    assert entry["triggered_by"] == "admin"
    assert entry["metrics"] == metrics


def test_multiple_entries_appended(temp_audit_log):
    import app.audit_log as audit_module
    audit_module.log_prediction_request("VH-001", "127.0.0.1", "c1", True)
    audit_module.log_prediction_request("VH-002", "127.0.0.1", "c2", False)
    lines = [l for l in temp_audit_log.read_text().strip().split("\n") if l]
    assert len(lines) == 2


def test_recent_audit_entries_returns_list(temp_audit_log):
    import app.audit_log as audit_module
    audit_module.log_prediction_request("VH-001", "127.0.0.1", "c1", True)
    entries = audit_module.recent_audit_entries(n=10)
    assert isinstance(entries, list)


def test_recent_audit_entries_empty_when_no_log(tmp_path, monkeypatch):
    import app.audit_log as audit_module
    orig = audit_module._AUDIT_LOG_PATH
    audit_module._AUDIT_LOG_PATH = tmp_path / "nonexistent.jsonl"
    entries = audit_module.recent_audit_entries()
    audit_module._AUDIT_LOG_PATH = orig
    assert entries == []
