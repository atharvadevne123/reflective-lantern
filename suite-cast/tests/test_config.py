"""Tests for suite-cast app/config.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import Settings, settings


class TestSettings:
    def test_default_database_url(self) -> None:
        s = Settings()
        assert "sqlite" in s.database_url or s.database_url.startswith("postgres")

    def test_default_model_version(self) -> None:
        s = Settings()
        assert isinstance(s.model_version, str)
        assert len(s.model_version) > 0

    def test_default_base_room_rate(self) -> None:
        s = Settings()
        assert s.base_room_rate > 0

    def test_default_rate_limit(self) -> None:
        s = Settings()
        assert s.rate_limit_per_minute > 0

    def test_environment_override(self) -> None:
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            s = Settings()
            assert s.environment == "production"

    def test_log_level_override(self) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            s = Settings()
            assert s.log_level == "DEBUG"

    def test_api_prefix(self) -> None:
        s = Settings()
        assert s.api_prefix.startswith("/")

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_valid_log_levels(self, level: str) -> None:
        with patch.dict(os.environ, {"LOG_LEVEL": level}):
            s = Settings()
            assert s.log_level == level


class TestGlobalSettings:
    def test_settings_is_settings_instance(self) -> None:
        assert isinstance(settings, Settings)

    def test_settings_has_database_url(self) -> None:
        assert hasattr(settings, "database_url")
        assert settings.database_url
