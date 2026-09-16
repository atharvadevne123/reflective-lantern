"""Logging configuration tests for Ticket-Oracle."""

from __future__ import annotations

import json
import logging

import pytest

from app.logging_config import _JsonFormatter, configure_logging


def test_json_formatter_emits_valid_json():
    formatter = _JsonFormatter()
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
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed


def test_json_formatter_includes_extras():
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="with extra",
        args=(),
        exc_info=None,
    )
    record.ticket_id = "T-999"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["ticket_id"] == "T-999"


def test_configure_logging_sets_level():
    configure_logging(level="DEBUG", fmt="text")
    root = logging.getLogger()
    assert root.level == logging.DEBUG


def test_configure_logging_json_handler():
    configure_logging(level="INFO", fmt="json")
    root = logging.getLogger()
    assert root.handlers
    assert isinstance(root.handlers[0].formatter, _JsonFormatter)


def test_configure_logging_clears_existing_handlers():
    configure_logging(level="INFO", fmt="text")
    configure_logging(level="INFO", fmt="text")
    root = logging.getLogger()
    assert len(root.handlers) == 1
