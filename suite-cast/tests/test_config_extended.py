"""Extended tests for suite-cast app/config.py Settings dataclass."""

from __future__ import annotations

import pytest

from app.config import Settings


class TestSettingsDefaults:
    def test_api_prefix_default(self) -> None:
        s = Settings()
        assert s.api_prefix == "/api/v1"

    def test_model_version_default(self) -> None:
        s = Settings()
        assert s.model_version == "1.0.0"

    def test_base_room_rate_positive(self) -> None:
        s = Settings()
        assert s.base_room_rate > 0.0

    def test_rate_limit_positive(self) -> None:
        s = Settings()
        assert s.rate_limit_per_minute > 0

    def test_log_level_string(self) -> None:
        s = Settings()
        assert isinstance(s.log_level, str)
        assert len(s.log_level) > 0

    def test_environment_string(self) -> None:
        s = Settings()
        assert isinstance(s.environment, str)


class TestSettingsFromEnv:
    def test_reads_model_version_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("MODEL_VERSION", "2.5.0")
        s = Settings()
        assert s.model_version == "2.5.0"

    def test_reads_base_room_rate_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("BASE_ROOM_RATE", "200.0")
        s = Settings()
        assert s.base_room_rate == pytest.approx(200.0)

    def test_reads_rate_limit_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "120")
        s = Settings()
        assert s.rate_limit_per_minute == 120

    def test_reads_log_level_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings()
        assert s.log_level == "DEBUG"

    def test_reads_environment_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        s = Settings()
        assert s.environment == "production"

    def test_reads_database_url_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./custom.db")
        s = Settings()
        assert "custom.db" in s.database_url
