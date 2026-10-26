"""Tests for structured JSON logging."""

import json
import logging

from app.logging_config import JsonFormatter, configure_logging


def _record(**kwargs) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=kwargs.pop("msg", "hello %s"),
        args=kwargs.pop("args", ("world",)),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_formats_valid_json() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_includes_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record(correlation_id="abc-123")))
    assert payload["correlation_id"] == "abc-123"


def test_omits_reserved_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record()))
    for reserved in ("pathname", "lineno", "msg", "args"):
        assert reserved not in payload


def test_serialises_non_json_values() -> None:
    """Non-serialisable extras must not blow up the formatter."""
    payload = json.loads(JsonFormatter().format(_record(obj=object())))
    assert isinstance(payload["obj"], str)


def test_includes_exception_text() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_sets_level() -> None:
    configure_logging(level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    configure_logging(level="INFO")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_unknown_level_defaults_to_info() -> None:
    configure_logging(level="NONSENSE")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_json_mode_installs_formatter() -> None:
    configure_logging(level="INFO", json_output=True)
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)
    configure_logging(level="INFO")
