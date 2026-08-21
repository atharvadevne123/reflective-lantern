"""Structured JSON logging configuration for Watt-Guard."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON lines for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a log record to a single JSON line.

        Args:
            record: The log record to format.

        Returns:
            JSON-encoded string with ts, level, logger, msg and optional fields.
        """
        log: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log["request_id"] = record.request_id
        if hasattr(record, "correlation_id"):
            log["correlation_id"] = record.correlation_id
        return json.dumps(log, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Set up root logger with optional JSON formatting.

    Args:
        level: Logging level name (e.g. "INFO", "DEBUG").
        json_output: Emit structured JSON lines when True; plain text otherwise.
    """
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


class TraceIdFilter(logging.Filter):
    """Inject a trace_id field into every log record that lacks one."""

    def __init__(self, trace_id: str = "") -> None:
        super().__init__()
        self.trace_id = trace_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = self.trace_id  # type: ignore[attr-defined]
        return True


def add_trace_id_filter(logger: logging.Logger, trace_id: str = "") -> None:
    """Attach a TraceIdFilter to *logger* so every record carries a trace_id.

    Args:
        logger: The logger instance to configure.
        trace_id: The trace identifier to stamp on records (e.g. a request UUID).
    """
    for handler in logger.handlers:
        handler.addFilter(TraceIdFilter(trace_id=trace_id))
    if not logger.handlers:
        logger.addFilter(TraceIdFilter(trace_id=trace_id))


def log_level_int(level_name: str) -> int:
    """Convert a log level name to its integer value.

    Args:
        level_name: Log level string such as 'DEBUG', 'INFO', 'WARNING'.

    Returns:
        Integer log level (e.g. 10 for DEBUG, 20 for INFO).

    Raises:
        ValueError: If the level name is not recognised by the logging module.
    """
    numeric = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level: {level_name!r}")
    return numeric


def get_configured_log_level() -> int:
    """Return the root logger's current effective log level as an integer.

    Returns:
        Current effective log level integer (e.g. 20 for INFO).
    """
    return logging.getLogger().getEffectiveLevel()


def is_debug_enabled() -> bool:
    """Return True if DEBUG logging is enabled for the root logger."""
    return logging.getLogger().isEnabledFor(logging.DEBUG)


__all__ = [
    "JsonFormatter",
    "TraceIdFilter",
    "add_trace_id_filter",
    "configure_logging",
    "filter",
    "format",
    "get_configured_log_level",
    "is_debug_enabled",
    "log_level_int",
]


def suppress_noisy_loggers(*names: str, level: str = "WARNING") -> None:
    """Set *level* on each named logger to reduce log noise in production.

    A common pattern for silencing verbose third-party loggers (e.g. boto3,
    urllib3) without touching the root logger.

    Args:
        *names: Logger names to silence (e.g. "boto3", "urllib3").
        level: Logging level to set on each logger. Defaults to "WARNING".
    """
    int_level = log_level_int(level)
    for name in names:
        logging.getLogger(name).setLevel(int_level)


def log_handler_count(logger_name: str = "root") -> int:
    """Return the number of handlers attached to the named logger.

    Args:
        logger_name: Logger name; 'root' returns the root logger's handler count.

    Returns:
        Handler count as a non-negative integer.
    """
    log = logging.getLogger() if logger_name == "root" else logging.getLogger(logger_name)
    return len(log.handlers)


def has_console_handler(logger_name: str = "root") -> bool:
    """Return True if the named logger has a StreamHandler (console) attached.

    Args:
        logger_name: Logger name; 'root' checks the root logger.

    Returns:
        True if at least one :class:`logging.StreamHandler` is present.
    """
    log = logging.getLogger() if logger_name == "root" else logging.getLogger(logger_name)
    return any(isinstance(h, logging.StreamHandler) for h in log.handlers)


def log_level_name(level_int: int) -> str:
    """Return the human-readable name for a numeric logging level.

    Args:
        level_int: Numeric level (e.g. 10 for DEBUG, 20 for INFO).

    Returns:
        Level name string (e.g. 'DEBUG'); 'UNKNOWN' for unregistered levels.
    """
    name = logging.getLevelName(level_int)
    if isinstance(name, str) and not name.startswith("Level "):
        return name
    return "UNKNOWN"


def has_handler(logger_name: str = "root") -> bool:
    """Return True if the named logger currently has any handlers attached.

    Args:
        logger_name: Logger name; use ``"root"`` for the root logger.

    Returns:
        True when at least one handler is registered.
    """
    logger_obj = logging.getLogger() if logger_name == "root" else logging.getLogger(logger_name)
    return len(logger_obj.handlers) > 0


def clear_handlers(logger_name: str = "root") -> int:
    """Remove all handlers from the named logger and return the count removed.

    Args:
        logger_name: Logger name; use ``"root"`` for the root logger.

    Returns:
        Number of handlers removed.
    """
    logger_obj = logging.getLogger() if logger_name == "root" else logging.getLogger(logger_name)
    count = len(logger_obj.handlers)
    for handler in list(logger_obj.handlers):
        logger_obj.removeHandler(handler)
    return count
