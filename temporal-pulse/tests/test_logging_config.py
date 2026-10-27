"""Tests for temporal-pulse app/logging_config.py."""

from __future__ import annotations

import json
import logging

import pytest

from app.logging_config import JsonFormatter, configure_logging


class TestJsonFormatter:
    def test_format_returns_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_contains_level(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 1, "msg", (), None)
        data = json.loads(formatter.format(record))
        assert "level" in data

    def test_contains_message(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 1, "test message", (), None)
        data = json.loads(formatter.format(record))
        assert "message" in data
        assert "test message" in data["message"]

    def test_contains_timestamp(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord("t", logging.INFO, "", 1, "msg", (), None)
        data = json.loads(formatter.format(record))
        assert "timestamp" in data


class TestConfigureLogging:
    def test_does_not_raise(self) -> None:
        configure_logging("INFO")

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_various_levels(self, level: str) -> None:
        configure_logging(level)
        root = logging.getLogger()
        assert root.level in range(logging.CRITICAL + 1)

    def test_json_format_flag(self) -> None:
        configure_logging("INFO", json_format=True)
        configure_logging("INFO", json_format=False)
