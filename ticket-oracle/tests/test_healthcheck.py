"""Healthcheck helper tests for Ticket-Oracle."""

from __future__ import annotations

import os
from unittest.mock import patch


def test_check_model_ready_false_when_no_files(tmp_path):
    with patch("app.model.MODEL_DIR", tmp_path):
        from app.healthcheck import check_model_ready
        result = check_model_ready()
    assert result["ready"] is False
    assert result["priority_model_exists"] is False
    assert result["sla_model_exists"] is False


def test_check_model_ready_true_when_files_exist(tmp_path):
    (tmp_path / "priority_model.joblib").write_bytes(b"fake")
    (tmp_path / "sla_model.joblib").write_bytes(b"fake")

    with patch("app.model.MODEL_DIR", tmp_path):
        from app.healthcheck import check_model_ready
        result = check_model_ready()
    assert result["ready"] is True


def test_check_db_reachable_sqlite(tmp_path):
    db_path = tmp_path / "test_health.db"
    db_url = f"sqlite:///{db_path}"

    with patch.dict(os.environ, {"DATABASE_URL": db_url}):
        import app.database as db_mod
        # Reset engine so it picks up new URL
        db_mod._engine = None
        db_mod._SessionLocal = None
        db_mod.DATABASE_URL = db_url

        from app.healthcheck import check_db_reachable
        result = check_db_reachable()

    assert result["reachable"] is True
    assert result["latency_ms"] >= 0


def test_full_health_has_required_keys(tmp_path):
    (tmp_path / "priority_model.joblib").write_bytes(b"x")
    (tmp_path / "sla_model.joblib").write_bytes(b"x")

    with patch("app.model.MODEL_DIR", tmp_path):
        from app.healthcheck import full_health
        result = full_health()

    assert "status" in result
    assert "model" in result
    assert "database" in result
    assert result["status"] in ("ok", "degraded")
