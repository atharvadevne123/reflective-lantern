"""Tests for config.logging_config."""

from __future__ import annotations

import json
import logging

import pytest


def test_configure_logging_sets_level() -> None:
    from config.logging_config import configure_logging
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    configure_logging("INFO")  # restore default


def test_configure_logging_default_is_info() -> None:
    from config.logging_config import configure_logging
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_adds_handler() -> None:
    from config.logging_config import configure_logging
    configure_logging()
    root = logging.getLogger()
    assert len(root.handlers) >= 1


def test_json_formatter_output_is_valid_json() -> None:
    from config.logging_config import JsonFormatter
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello world", args=(), exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed
    assert "logger" in parsed


def test_json_formatter_includes_exception() -> None:
    from config.logging_config import JsonFormatter
    formatter = JsonFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=0,
        msg="oops", args=(), exc_info=exc_info,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" in parsed
    assert "ValueError" in parsed["exception"]


def test_configure_logging_json_mode() -> None:
    from config.logging_config import JsonFormatter, configure_logging
    configure_logging(json_logs=True)
    root = logging.getLogger()
    assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)
    configure_logging()  # restore


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_configure_logging_all_levels(level: str) -> None:
    from config.logging_config import configure_logging
    configure_logging(level)
    assert logging.getLogger().level == getattr(logging, level)
    configure_logging("INFO")  # restore
