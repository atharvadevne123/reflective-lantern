"""Tests for health check utilities."""
from __future__ import annotations

import tempfile

from app.health import (
    HealthStatus,
    build_health_status,
    check_database,
    check_model_file,
)


def test_check_database_ok():
    assert check_database("sqlite:///test.db") == "ok"


def test_check_database_misconfigured():
    assert check_database("") == "misconfigured"


def test_check_model_file_missing():
    assert check_model_file("/nonexistent/path/model.pkl") == "missing"


def test_check_model_file_loaded():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        assert check_model_file(f.name) == "loaded"


def test_build_health_status_healthy():
    with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
        hs = build_health_status("sqlite:///dev.db", f.name)
    assert hs.status == "healthy"
    assert hs.is_healthy()


def test_build_health_status_degraded_when_model_missing():
    hs = build_health_status("sqlite:///dev.db", "/nonexistent/model.pkl")
    assert hs.status == "degraded"
    assert not hs.is_healthy()


def test_build_health_status_checks_keys():
    hs = build_health_status("sqlite:///dev.db", "/nonexistent/model.pkl")
    assert "database" in hs.checks
    assert "model" in hs.checks


def test_health_status_to_dict():
    hs = HealthStatus(status="healthy", checks={"db": "ok"}, uptime_seconds=42.5)
    d = hs.to_dict()
    assert d["status"] == "healthy"
    assert d["checks"]["db"] == "ok"
    assert d["uptime_seconds"] == 42.5


def test_uptime_positive():
    hs = build_health_status("sqlite:///dev.db", "/nonexistent/model.pkl")
    assert hs.uptime_seconds >= 0.0
