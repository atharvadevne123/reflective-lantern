"""Tests for forge-guard app/logging_config.py."""

from __future__ import annotations

import json
import logging

import pytest

from app.logging_config import JsonFormatter, configure_logging


class TestJsonFormatter:
    def test_format_returns_string(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert isinstance(result, str)

    def test_format_is_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="", lineno=0,
            msg="test message", args=(), exc_info=None
        )
        data = json.loads(formatter.format(record))
        assert isinstance(data, dict)

    def test_format_has_required_keys(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="mylogger", level=logging.ERROR, pathname="", lineno=0,
            msg="error happened", args=(), exc_info=None
        )
        data = json.loads(formatter.format(record))
        assert "ts" in data
        assert "level" in data
        assert "message" in data
        assert "logger" in data

    def test_format_level_name(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="l", level=logging.DEBUG, pathname="", lineno=0,
            msg="dbg", args=(), exc_info=None
        )
        data = json.loads(formatter.format(record))
        assert data["level"] == "DEBUG"

    def test_format_message_content(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="l", level=logging.INFO, pathname="", lineno=0,
            msg="specific message here", args=(), exc_info=None
        )
        data = json.loads(formatter.format(record))
        assert "specific message here" in data["message"]


class TestConfigureLogging:
    def test_configure_plain(self) -> None:
        configure_logging(level="WARNING", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_configure_json(self) -> None:
        configure_logging(level="DEBUG", json_output=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        handler = root.handlers[-1]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_configure_removes_old_handlers(self) -> None:
        configure_logging(level="INFO", json_output=False)
        configure_logging(level="INFO", json_output=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
