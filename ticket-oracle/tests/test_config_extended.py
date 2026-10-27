"""Extended tests for ticket-oracle app/config.py."""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


class TestSettingsDefaults:
    def test_create_instance(self) -> None:
        assert Settings() is not None

    def test_default_log_level(self) -> None:
        s = Settings()
        assert s.log_level == "INFO"

    def test_default_database_url_sqlite(self) -> None:
        s = Settings()
        assert "sqlite" in s.database_url

    def test_default_max_batch_size(self) -> None:
        s = Settings()
        assert s.max_batch_size > 0

    def test_default_cache_ttl_positive(self) -> None:
        s = Settings()
        assert s.cache_ttl_seconds > 0.0

    def test_default_drift_window_positive(self) -> None:
        s = Settings()
        assert s.drift_window_hours > 0

    def test_default_spike_z_threshold_positive(self) -> None:
        s = Settings()
        assert s.spike_z_threshold > 0.0


class TestGetSettings:
    def test_returns_settings(self) -> None:
        s = get_settings()
        assert isinstance(s, Settings)

    def test_cached_returns_same_instance(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestSettingsEnvOverride:
    def test_log_level_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_max_batch_size_env(self, monkeypatch) -> None:
        monkeypatch.setenv("MAX_BATCH_SIZE", "50")
        s = Settings()
        assert s.max_batch_size == 50
