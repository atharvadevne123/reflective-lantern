"""Tests for the logging configuration module."""

from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging, get_logger, log_dict


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
