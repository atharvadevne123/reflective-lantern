"""Tests for app.config_schema module."""

from __future__ import annotations

import pytest

from app.config_schema import ConfigSchema, env_bool, env_float, env_int, load_from_env


class TestEnvBool:
    @pytest.mark.parametrize("val", ["1", "true", "True", "yes", "on"])
    def test_truthy_strings(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLAG", val)
        assert env_bool("TEST_FLAG") is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off"])
    def test_falsy_strings(self, val: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLAG", val)
        assert env_bool("TEST_FLAG") is False

    def test_missing_returns_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_FLAG", raising=False)
        assert env_bool("TEST_FLAG") is False


class TestEnvInt:
    def test_reads_integer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "42")
        assert env_int("TEST_INT") == 42

    def test_missing_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_INT", raising=False)
        assert env_int("TEST_INT", default=7) == 7


class TestConfigSchema:
    def test_default_values(self) -> None:
        cfg = ConfigSchema()
        assert cfg.debug is False
        assert cfg.log_level == "INFO"
        assert cfg.max_workers == 4

    def test_is_production_when_debug_false(self) -> None:
        assert ConfigSchema(debug=False).is_production() is True

    def test_is_not_production_when_debug_true(self) -> None:
        assert ConfigSchema(debug=True).is_production() is False


class TestLoadFromEnv:
    def test_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_DEBUG", "true")
        monkeypatch.setenv("APP_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("APP_MAX_WORKERS", "8")
        cfg = load_from_env()
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"
        assert cfg.max_workers == 8
