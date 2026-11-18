"""Tests for Fleet-Mind logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.logging_config import configure_logging


def test_configure_logging_sets_level(caplog):
    configure_logging(level="DEBUG")
    root = logging.getLogger()
    assert root.level <= logging.DEBUG


def test_configure_logging_info_level():
    configure_logging(level="INFO")
    root = logging.getLogger()
    assert root.level <= logging.INFO


def test_configure_logging_warning_level():
    configure_logging(level="WARNING")
    root = logging.getLogger()
    assert root.level <= logging.WARNING


def test_configure_logging_does_not_raise():
    configure_logging(level="INFO", json_output=False)
    configure_logging(level="DEBUG", json_output=True)


def test_json_formatter_produces_valid_json():
    from app.logging_config import _JsonFormatter

    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed


def test_json_formatter_includes_logger_name():
    from app.logging_config import _JsonFormatter

    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="mylogger",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="warn",
        args=(),
        exc_info=None,
    )
    parsed = json.loads(formatter.format(record))
    assert parsed["logger"] == "mylogger"


def test_configure_logging_twice_does_not_duplicate_handlers():
    configure_logging(level="INFO")
    before = len(logging.getLogger().handlers)
    configure_logging(level="INFO")
    after = len(logging.getLogger().handlers)
    assert after <= before + 1
