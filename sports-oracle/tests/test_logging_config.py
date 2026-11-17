"""Tests for Sports-Oracle logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logging_config import configure_logging


def test_configure_logging_does_not_raise():
    configure_logging(level="INFO")


def test_configure_logging_debug_level():
    configure_logging(level="DEBUG")
    root = logging.getLogger()
    assert root.level <= logging.DEBUG


def test_configure_logging_warning_level():
    configure_logging(level="WARNING")
    root = logging.getLogger()
    assert root.level <= logging.WARNING


def test_configure_logging_twice_is_safe():
    configure_logging(level="INFO")
    configure_logging(level="DEBUG")


def test_noisy_loggers_suppressed_after_configure():
    configure_logging(level="INFO")
    for name in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        level = logging.getLogger(name).level
        assert level >= logging.WARNING


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_all_log_levels_accepted(level):
    configure_logging(level=level)
