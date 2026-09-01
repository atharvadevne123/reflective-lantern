"""Tests for environment-driven configuration."""

import pytest

from app.config import Settings, get_settings


def test_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DATABASE_URL", "API_PORT", "DRIFT_THRESHOLD", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.database_url == "sqlite:///./threat_lens.db"
    assert s.api_port == 8000
    assert s.drift_threshold == 0.05
    assert s.rate_limit_per_minute == 120


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("DRIFT_THRESHOLD", "0.01")
    s = Settings()
    assert s.api_port == 9999
    assert s.drift_threshold == 0.01


def test_malformed_int_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad value must not crash startup — it falls back and warns."""
    monkeypatch.setenv("API_PORT", "not-a-port")
    assert Settings().api_port == 8000


def test_malformed_float_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIFT_THRESHOLD", "high")
    assert Settings().drift_threshold == 0.05


def test_is_postgres_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
    assert Settings().is_postgres() is True
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./local.db")
    assert Settings().is_postgres() is False


def test_get_settings_returns_settings() -> None:
    assert isinstance(get_settings(), Settings)


def test_settings_are_frozen() -> None:
    s = Settings()
    with pytest.raises(Exception):
        s.api_port = 1234  # type: ignore[misc]
