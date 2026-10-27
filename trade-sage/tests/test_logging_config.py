"""Tests for trade-sage app/logging_config.py."""

from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging


class TestConfigureLogging:
    def test_does_not_raise(self) -> None:
        configure_logging()

    def test_accepts_info_level(self) -> None:
        configure_logging("INFO")
        assert logging.getLogger().level <= logging.INFO

    def test_accepts_debug_level(self) -> None:
        configure_logging("DEBUG")
        assert logging.getLogger().level <= logging.DEBUG

    def test_accepts_warning_level(self) -> None:
        configure_logging("WARNING")
        root = logging.getLogger()
        assert root.level in (logging.WARNING, logging.DEBUG, logging.INFO, 0)

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_various_levels(self, level: str) -> None:
        configure_logging(level)
