"""Tests for structured JSON logging configuration."""

from __future__ import annotations

import json
import logging

from app.logging_config import JsonFormatter, configure_logging


def test_json_formatter_emits_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert "ts" in payload


def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="x.py",
        lineno=1,
        msg="predict",
        args=(),
        exc_info=None,
    )
    record.correlation_id = "cid-42"
    record.prediction = 1
    payload = json.loads(formatter.format(record))
    assert payload["correlation_id"] == "cid-42"
    assert payload["prediction"] == 1


def test_json_formatter_includes_exception():
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="x.py",
            lineno=1,
            msg="failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "exception" in payload
    assert "ValueError" in payload["exception"]


def test_configure_logging_plain_and_json():
    configure_logging(level="DEBUG", json_output=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert isinstance(root.handlers[0].formatter, JsonFormatter)

    configure_logging(level="INFO", json_output=False)
    assert root.level == logging.INFO
    assert not isinstance(root.handlers[0].formatter, JsonFormatter)
