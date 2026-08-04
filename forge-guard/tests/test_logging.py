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


def test_json_formatter_timestamp_is_iso():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="x.py", lineno=1,
        msg="ts check", args=(), exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    ts = payload["ts"]
    assert "T" in ts and (ts.endswith("Z") or "+" in ts or ts.endswith("+00:00"))


def test_json_formatter_level_names():
    formatter = JsonFormatter()
    for level_name, level_const in [("DEBUG", logging.DEBUG), ("WARNING", logging.WARNING),
                                    ("ERROR", logging.ERROR)]:
        record = logging.LogRecord(
            name="test", level=level_const, pathname="x.py", lineno=1,
            msg="msg", args=(), exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        assert payload["level"] == level_name


def test_configure_logging_clears_previous_handlers():
    configure_logging(level="INFO", json_output=False)
    first_count = len(logging.getLogger().handlers)
    configure_logging(level="WARNING", json_output=False)
    second_count = len(logging.getLogger().handlers)
    assert second_count == first_count  # No duplicate handlers


def test_json_formatter_model_version_field():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="x.py", lineno=1,
        msg="model log", args=(), exc_info=None,
    )
    record.model_version = "1.1.0"
    payload = json.loads(formatter.format(record))
    assert payload.get("model_version") == "1.1.0"


def test_configure_logging_json_output_sets_formatter():
    configure_logging(level="INFO", json_output=True)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_json_formatter_message_formatting():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="x.py", lineno=1,
        msg="value is %d and %.2f",
        args=(42, 3.14),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert "42" in payload["message"]
    assert "3.14" in payload["message"]
