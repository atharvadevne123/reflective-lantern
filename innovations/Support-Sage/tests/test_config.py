"""Tests for configuration loading."""

from app.config import get_settings


def test_default_settings():
    s = get_settings()
    assert s.database_url
    assert s.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")
    assert s.drift_window_days > 0
    assert 0 < s.drift_p_threshold < 1


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("DRIFT_WINDOW_DAYS", "14")
    s = get_settings()
    assert s.drift_window_days == 14
