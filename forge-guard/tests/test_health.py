"""Tests for the health check utilities in app/health.py."""

from __future__ import annotations


def test_check_model_loaded_returns_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
    from app import health

    result = health.check_model_loaded()
    assert result["model_file_exists"] is False
    assert result["ok"] is False


def test_check_model_loaded_returns_true_when_present(tmp_path, monkeypatch):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake model content")
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    from app import health

    result = health.check_model_loaded()
    assert result["model_file_exists"] is True
    assert result["model_file_size_bytes"] > 0
    assert result["ok"] is True


def test_check_model_loaded_false_for_empty_file(tmp_path, monkeypatch):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"")
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    from app import health

    result = health.check_model_loaded()
    assert result["ok"] is False


def test_check_database_reachable_sqlite():
    from app import health

    result = health.check_database_reachable("sqlite:///:memory:")
    assert result["reachable"] is True
    assert result["url_scheme"] == "sqlite"


def test_check_database_reachable_bad_url():
    from app import health

    result = health.check_database_reachable("postgresql://bad:bad@localhost:9999/nonexistent")
    assert result["reachable"] is False
    assert "error" in result


def test_composite_health_returns_degraded_with_bad_db(tmp_path, monkeypatch):
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"content")
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("DATABASE_URL", "postgresql://bad:bad@localhost:9999/bad")
    from app import health

    result = health.composite_health()
    assert result["status"] == "degraded"
    assert result["model"]["ok"] is True
    assert result["database"]["reachable"] is False


def test_composite_health_keys_present(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from app import health

    result = health.composite_health()
    assert "status" in result
    assert "model" in result
    assert "database" in result


def test_check_database_reachable_returns_url_scheme():
    from app import health

    result = health.check_database_reachable("sqlite:///:memory:")
    assert result["url_scheme"] == "sqlite"
