"""Tests for the application configuration module."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings, reset_settings


class TestSettings:
    def setup_method(self) -> None:
        reset_settings()

    def test_defaults_applied(self) -> None:
        s = Settings()
        assert "sqlite" in s.database_url
        assert s.model_version == "1.0.0"
        assert s.log_level == "INFO"
        assert s.rate_limit == 60
        assert s.rate_window == 60

    def test_env_override_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        reset_settings()
        s = Settings()
        assert s.database_url == "postgresql://localhost/test"

    def test_env_override_model_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_VERSION", "2.5.0")
        reset_settings()
        s = Settings()
        assert s.model_version == "2.5.0"

    def test_log_level_uppercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        reset_settings()
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_rate_limit_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RATE_LIMIT", "120")
        reset_settings()
        s = Settings()
        assert s.rate_limit == 120


class TestGetSettings:
    def setup_method(self) -> None:
        reset_settings()

    def test_returns_settings_instance(self) -> None:
        assert isinstance(get_settings(), Settings)

    def test_singleton_behaviour(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_clears_singleton(self) -> None:
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2


@pytest.mark.parametrize("threshold", ["0.01", "0.05", "0.10"])
def test_drift_threshold_from_env(monkeypatch: pytest.MonkeyPatch, threshold: str) -> None:
    monkeypatch.setenv("DRIFT_THRESHOLD", threshold)
    reset_settings()
    s = Settings()
    assert abs(s.drift_threshold - float(threshold)) < 1e-9
    reset_settings()
