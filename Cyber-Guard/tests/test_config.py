"""Configuration loading tests for Cyber-Guard."""

from __future__ import annotations

import pytest

from app.config import Settings, _env_float, _env_int, get_settings


def test_defaults_when_env_unset(monkeypatch):
    for var in ("DRIFT_P_THRESHOLD", "REFERENCE_WINDOW_DAYS", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(var, raising=False)
    s = get_settings()
    assert s.drift_p_threshold == 0.05
    assert s.reference_window_days == 7
    assert s.rate_limit_per_minute == 120


def test_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv("DRIFT_P_THRESHOLD", "0.01")
    monkeypatch.setenv("REFERENCE_WINDOW_DAYS", "30")
    s = get_settings()
    assert s.drift_p_threshold == 0.01
    assert s.reference_window_days == 30


def test_malformed_float_falls_back(monkeypatch):
    """A bad env value must not crash startup -- it falls back to the default."""
    monkeypatch.setenv("DRIFT_P_THRESHOLD", "not-a-number")
    assert _env_float("DRIFT_P_THRESHOLD", 0.05) == 0.05


def test_malformed_int_falls_back(monkeypatch):
    monkeypatch.setenv("REFERENCE_WINDOW_DAYS", "seven")
    assert _env_int("REFERENCE_WINDOW_DAYS", 7) == 7


@pytest.mark.parametrize("url,expected", [
    ("sqlite:///./cyber_guard.db", True),
    ("postgresql://u:p@localhost:5432/db", False),
])
def test_is_sqlite_detection(url: str, expected: bool, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    assert get_settings().is_sqlite() is expected


def test_settings_are_frozen():
    """Settings is a snapshot; mutating it at runtime must fail loudly."""
    s = Settings()
    with pytest.raises(Exception):
        s.database_url = "postgresql://elsewhere/db"
