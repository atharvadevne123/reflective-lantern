"""Tests for structured JSON logging configuration."""

from __future__ import annotations

import json
import logging
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestJsonFormatter:
    def test_output_is_valid_json(self):
        from app.logging_config import JsonFormatter
        record = logging.LogRecord("test", logging.INFO, "f.py", 1, "hello", None, None)
        line = JsonFormatter().format(record)
        payload = json.loads(line)
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"

    def test_includes_exception(self):
        from app.logging_config import JsonFormatter
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                "test", logging.ERROR, "f.py", 1, "failed", None, sys.exc_info()
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "exception" in payload
        assert "boom" in payload["exception"]

    def test_includes_correlation_id_extra(self):
        from app.logging_config import JsonFormatter
        record = logging.LogRecord("test", logging.INFO, "f.py", 1, "msg", None, None)
        record.correlation_id = "abc-123"
        payload = json.loads(JsonFormatter().format(record))
        assert payload["correlation_id"] == "abc-123"


class TestConfigureLogging:
    def test_sets_level(self):
        from app.logging_config import configure_logging
        configure_logging(level="DEBUG", json_format=False)
        assert logging.getLogger().level == logging.DEBUG
        configure_logging(level="INFO", json_format=False)

    def test_json_format_flag(self):
        from app.logging_config import configure_logging, JsonFormatter
        configure_logging(level="INFO", json_format=True)
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)
        configure_logging(level="INFO", json_format=False)

    def test_quiets_noisy_loggers(self):
        from app.logging_config import configure_logging
        configure_logging(level="DEBUG", json_format=False)
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING
        configure_logging(level="INFO", json_format=False)
