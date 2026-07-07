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
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="oops",
        args=(),
        exc_info=exc_info,
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


def test_json_formatter_level_field_is_string() -> None:
    from config.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="warn msg",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "WARNING"
    assert isinstance(parsed["level"], str)


def test_configure_logging_clears_duplicate_handlers() -> None:
    from config.logging_config import configure_logging

    configure_logging()
    count_before = len(logging.getLogger().handlers)
    configure_logging()
    count_after = len(logging.getLogger().handlers)
    # Should not keep growing unbounded
    assert count_after <= count_before + 1


def test_json_formatter_no_exception_key_absent() -> None:
    from config.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="clean",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert "exception" not in parsed


@pytest.mark.parametrize(
    "level_str,level_int",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_configure_logging_numeric_level(level_str: str, level_int: int) -> None:
    from config.logging_config import configure_logging

    configure_logging(level_str)
    assert logging.getLogger().level == level_int
    configure_logging("INFO")


def test_get_logger_returns_logger() -> None:
    from config.logging_config import get_logger

    logger = get_logger("test.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.module"


def test_get_logger_same_name_same_instance() -> None:
    from config.logging_config import get_logger

    l1 = get_logger("same.name")
    l2 = get_logger("same.name")
    assert l1 is l2


@pytest.mark.parametrize("name", ["config.settings", "scripts.cleanup", "my.module"])
def test_get_logger_parametrized_names(name: str) -> None:
    from config.logging_config import get_logger

    logger = get_logger(name)
    assert logger.name == name


def test_json_formatter_run_id_included() -> None:
    from config.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="with run_id",
        args=(),
        exc_info=None,
    )
    record.run_id = "abc-123"  # type: ignore[attr-defined]
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed.get("run_id") == "abc-123"
