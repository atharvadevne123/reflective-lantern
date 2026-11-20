"""Tests for the logging configuration module."""

from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging, get_logger, is_valid_level, log_dict


class TestConfigureLogging:
    def test_sets_root_level(self) -> None:
        configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_info_level(self) -> None:
        configure_logging(level="INFO")
        assert logging.getLogger().level == logging.INFO

    def test_warning_level(self) -> None:
        configure_logging(level="WARNING")
        assert logging.getLogger().level == logging.WARNING


class TestGetLogger:
    def test_returns_logger(self) -> None:
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self) -> None:
        logger = get_logger("social_pulse.test")
        assert logger.name == "social_pulse.test"

    def test_different_names_return_different_loggers(self) -> None:
        l1 = get_logger("a.module")
        l2 = get_logger("b.module")
        assert l1 is not l2


class TestLogDict:
    def test_does_not_raise(self, caplog) -> None:
        logger = get_logger("test")
        with caplog.at_level(logging.INFO):
            log_dict(logger, logging.INFO, "test message", {"key": "value", "count": 42})

    def test_message_in_log(self, caplog) -> None:
        logger = get_logger("test.log_dict")
        with caplog.at_level(logging.INFO):
            log_dict(logger, logging.INFO, "prediction", {"p": 0.75, "label": True})
        assert any("prediction" in r.message for r in caplog.records)


@pytest.mark.parametrize("level_str", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_configure_logging_accepts_all_levels(level_str: str) -> None:
    configure_logging(level=level_str)
    expected = getattr(logging, level_str)
    assert logging.getLogger().level == expected


class TestIsValidLevel:
    def test_debug_is_valid(self) -> None:
        assert is_valid_level("DEBUG") is True

    def test_info_is_valid(self) -> None:
        assert is_valid_level("INFO") is True

    def test_warning_is_valid(self) -> None:
        assert is_valid_level("WARNING") is True

    def test_error_is_valid(self) -> None:
        assert is_valid_level("ERROR") is True

    def test_critical_is_valid(self) -> None:
        assert is_valid_level("CRITICAL") is True

    def test_lowercase_accepted(self) -> None:
        assert is_valid_level("debug") is True

    def test_mixed_case_accepted(self) -> None:
        assert is_valid_level("Warning") is True

    def test_invalid_returns_false(self) -> None:
        assert is_valid_level("VERBOSE") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_valid_level("") is False

    def test_numeric_string_returns_false(self) -> None:
        assert is_valid_level("20") is False


@pytest.mark.parametrize(
    "level,expected",
    [
        ("DEBUG", True),
        ("INFO", True),
        ("CRITICAL", True),
        ("TRACE", False),
        ("ALL", False),
    ],
)
def test_is_valid_level_parametrized(level: str, expected: bool) -> None:
    assert is_valid_level(level) == expected
