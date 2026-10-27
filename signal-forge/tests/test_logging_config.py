"""Tests for signal-forge app/logging_config.py."""

from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging, get_logging_config


class TestGetLoggingConfig:
    def test_returns_dict(self) -> None:
        config = get_logging_config()
        assert isinstance(config, dict)

    def test_has_version_key(self) -> None:
        config = get_logging_config()
        assert config.get("version") == 1

    def test_has_formatters(self) -> None:
        config = get_logging_config()
        assert "formatters" in config

    def test_has_handlers(self) -> None:
        config = get_logging_config()
        assert "handlers" in config

    def test_has_root_logger(self) -> None:
        config = get_logging_config()
        assert "root" in config

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_level_in_root(self, level: str) -> None:
        config = get_logging_config(log_level=level)
        assert config["root"]["level"] == level


class TestConfigureLogging:
    def test_does_not_raise(self) -> None:
        configure_logging("INFO")

    def test_sets_log_level(self) -> None:
        configure_logging("WARNING")
        root = logging.getLogger()
        assert root.level <= logging.WARNING

    def test_debug_level(self) -> None:
        configure_logging("DEBUG")
        root = logging.getLogger()
        assert root.level <= logging.DEBUG
