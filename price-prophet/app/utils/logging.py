"""
Logging configuration and utilities for Price-Prophet.

Provides a one-call setup helper, a named-logger factory, and a
function-call tracing decorator that integrates with the standard
library :mod:`logging` module.
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable

_DEFAULT_FMT = "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
_DEFAULT_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    fmt: str | None = None,
) -> None:
    """Configure the root logger with a :class:`~logging.StreamHandler`.

    Safe to call multiple times — duplicate handlers are avoided by
    checking whether the root logger already has handlers attached.

    Parameters
    ----------
    level:
        Log level string, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
        Defaults to ``"INFO"``.
    fmt:
        Optional custom format string passed to
        :class:`~logging.Formatter`.  Defaults to a timestamped format
        showing level, logger name, and message.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt or _DEFAULT_FMT,
        datefmt=_DEFAULT_DATE_FMT,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    if not root.handlers:
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`~logging.Logger`.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


def log_call(func: Callable) -> Callable:
    """Decorator that logs function entry and exit at DEBUG level.

    On entry the decorator logs the function name and its positional /
    keyword arguments.  On exit it logs the return value (repr-truncated
    to 200 characters).  Exceptions are re-raised after logging.

    Parameters
    ----------
    func:
        Function to wrap.

    Returns
    -------
    Callable
        Wrapped function with identical signature.

    Examples
    --------
    >>> @log_call
    ... def add(a, b):
    ...     return a + b
    """
    _logger = logging.getLogger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[return]
        """Log entry and exit of the wrapped function at DEBUG level."""
        _logger.debug(
            "CALL %s args=%r kwargs=%r",
            func.__qualname__,
            args,
            kwargs,
        )
        try:
            result = func(*args, **kwargs)
            result_repr = repr(result)
            if len(result_repr) > 200:
                result_repr = result_repr[:197] + "..."
            _logger.debug("RETURN %s -> %s", func.__qualname__, result_repr)
            return result
        except Exception as exc:
            _logger.debug(
                "EXCEPTION %s raised %s: %s",
                func.__qualname__,
                type(exc).__name__,
                exc,
            )
            raise

    return wrapper
