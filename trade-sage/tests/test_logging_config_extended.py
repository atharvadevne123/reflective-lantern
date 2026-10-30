"""Extended tests for logging configuration utilities."""

from __future__ import annotations

import logging

import pytest

from app.logging_config import get_logger, silence_noisy_loggers


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        result = get_logger("test.module")
        assert isinstance(result, logging.Logger)

    def test_logger_has_correct_name(self) -> None:
        logger = get_logger("my.test.logger")
        assert logger.name == "my.test.logger"

    def test_level_override_applied(self) -> None:
        logger = get_logger("level.test", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_warning_level_set(self) -> None:
        logger = get_logger("warn.test", level="WARNING")
        assert logger.level == logging.WARNING

    def test_no_level_returns_logger(self) -> None:
        logger = get_logger("no.level.test")
        assert logger is not None

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_standard_levels(self, level: str) -> None:
        logger = get_logger(f"level.{level.lower()}", level=level)
        assert logger.level == getattr(logging, level)


class TestSilenceNoisyLoggers:
    def test_sets_warning_level(self) -> None:
        silence_noisy_loggers(["test.noisy.logger"])
        assert logging.getLogger("test.noisy.logger").level == logging.WARNING

    def test_sets_custom_level(self) -> None:
        silence_noisy_loggers(["test.error.logger"], level="ERROR")
        assert logging.getLogger("test.error.logger").level == logging.ERROR

    def test_empty_list_no_error(self) -> None:
        silence_noisy_loggers([])

    def test_multiple_loggers(self) -> None:
        loggers = ["test.multi.a", "test.multi.b", "test.multi.c"]
        silence_noisy_loggers(loggers, level="WARNING")
        for name in loggers:
            assert logging.getLogger(name).level == logging.WARNING
