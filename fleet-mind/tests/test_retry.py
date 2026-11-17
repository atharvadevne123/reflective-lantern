"""Tests for Fleet-Mind retry decorator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retry import retry


def test_retry_succeeds_on_first_attempt():
    counter = {"calls": 0}

    @retry(max_attempts=3)
    def succeeds():
        counter["calls"] += 1
        return "ok"

    result = succeeds()
    assert result == "ok"
    assert counter["calls"] == 1


def test_retry_retries_on_failure_then_succeeds():
    counter = {"calls": 0}

    @retry(max_attempts=3, base_delay=0.0)
    def flaky():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise ValueError("not yet")
        return "done"

    result = flaky()
    assert result == "done"
    assert counter["calls"] == 3


def test_retry_raises_after_max_attempts():
    @retry(max_attempts=3, base_delay=0.0)
    def always_fails():
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        always_fails()


def test_retry_only_catches_specified_exceptions():
    @retry(max_attempts=3, base_delay=0.0, exceptions=(ValueError,))
    def raises_type_error():
        raise TypeError("wrong type")

    with pytest.raises(TypeError):
        raises_type_error()


def test_retry_catches_specified_exception():
    counter = {"calls": 0}

    @retry(max_attempts=3, base_delay=0.0, exceptions=(ValueError,))
    def raises_value_error_twice():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise ValueError("not yet")
        return "ok"

    result = raises_value_error_twice()
    assert result == "ok"


def test_retry_preserves_function_name():
    @retry(max_attempts=2, base_delay=0.0)
    def my_function():
        return 42

    assert my_function.__name__ == "my_function"


def test_retry_passes_args_and_kwargs():
    @retry(max_attempts=1)
    def add(a, b, c=0):
        return a + b + c

    assert add(1, 2, c=3) == 6


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_retry_exact_attempt_count_on_failure(max_attempts):
    counter = {"calls": 0}

    @retry(max_attempts=max_attempts, base_delay=0.0)
    def always_fails():
        counter["calls"] += 1
        raise OSError("fail")

    with pytest.raises(OSError):
        always_fails()

    assert counter["calls"] == max_attempts
