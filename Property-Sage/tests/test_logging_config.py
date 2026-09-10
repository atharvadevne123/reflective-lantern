"""Tests for the structured logging configuration."""

import json
import logging

from app.logging_config import JsonFormatter, configure_logging


def test_json_formatter_produces_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Hello %s",
        args=("world",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "Hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter(extra_fields={"service": "property-sage", "env": "test"})
    record = logging.LogRecord(
        name="x",
        level=logging.DEBUG,
        pathname="x.py",
        lineno=2,
        msg="test",
        args=(),
        exc_info=None,
    )
    parsed = json.loads(formatter.format(record))
    assert parsed["service"] == "property-sage"
    assert parsed["env"] == "test"


def test_configure_logging_sets_level():
    configure_logging(level="DEBUG", json_format=False)
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(level="INFO", json_format=False)


def test_configure_logging_json_mode():
    configure_logging(level="WARNING", json_format=True, service="test-svc")
    root = logging.getLogger()
    handler = root.handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)
    configure_logging(level="INFO", json_format=False)


def test_json_formatter_timestamp_present():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="ts",
        level=logging.ERROR,
        pathname="ts.py",
        lineno=5,
        msg="boom",
        args=(),
        exc_info=None,
    )
    parsed = json.loads(formatter.format(record))
    assert "timestamp" in parsed
    assert "T" in parsed["timestamp"]
