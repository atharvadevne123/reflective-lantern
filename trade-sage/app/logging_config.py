"""Structured JSON logging configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging for the application.

    Args:
        level: Logging level string (default INFO).
    """
    fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","msg":%(message)s,"module":"%(module)s"}'
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a named logger, optionally overriding the log level.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Optional level string to set on this specific logger.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def silence_noisy_loggers(names: list[str], level: str = "WARNING") -> None:
    """Set a higher log level on chatty third-party loggers.

    Args:
        names: List of logger names to silence.
        level: Level to apply (default WARNING).
    """
    log_level = getattr(logging, level.upper(), logging.WARNING)
    for name in names:
        logging.getLogger(name).setLevel(log_level)
