"""Extended tests for threat-lens app/logging_config.py."""

from __future__ import annotations

import json
import logging

import pytest

from app.logging_config import JsonFormatter, configure_logging


class TestJsonFormatterExtended:
    def test_format_has_timestamp(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("l", logging.INFO, "", 0, "msg", (), None)
        data = json.loads(formatter.format(record))
        assert "timestamp" in data

    def test_format_has_logger_name(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("my.logger", logging.INFO, "", 0, "hello", (), None)
        data = json.loads(formatter.format(record))
        assert data["logger"] == "my.logger"

    def test_format_extra_fields(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("l", logging.INFO, "", 0, "msg", (), None)
        record.custom_field = "custom_value"
        data = json.loads(formatter.format(record))
        assert data.get("custom_field") == "custom_value"

    def test_format_with_exception(self) -> None:
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        record = logging.LogRecord("l", logging.ERROR, "", 0, "err", (), exc_info)
        data = json.loads(formatter.format(record))
        assert "exception" in data

    def test_all_levels_parseable(self) -> None:
        formatter = JsonFormatter()
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR):
            record = logging.LogRecord("l", level, "", 0, "msg", (), None)
            result = formatter.format(record)
            data = json.loads(result)
            assert data["level"] == logging.getLevelName(level)


class TestConfigureLoggingExtended:
    def test_sets_root_level_debug(self) -> None:
        configure_logging(level="DEBUG", json_output=False)
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_root_level_error(self) -> None:
        configure_logging(level="ERROR", json_output=True)
        assert logging.getLogger().level == logging.ERROR

    def test_json_output_uses_json_formatter(self) -> None:
        configure_logging(level="INFO", json_output=True)
        handler = logging.getLogger().handlers[-1]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_plain_output_does_not_use_json_formatter(self) -> None:
        configure_logging(level="INFO", json_output=False)
        handler = logging.getLogger().handlers[-1]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_replaces_existing_handlers(self) -> None:
        configure_logging(level="INFO", json_output=False)
        configure_logging(level="INFO", json_output=False)
        assert len(logging.getLogger().handlers) == 1
