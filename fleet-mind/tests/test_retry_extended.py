"""Extended tests for Fleet-Mind retry decorator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retry import retry


def test_retry_zero_delay_succeeds_first_try():
    calls = []

    @retry(max_attempts=3, base_delay=0.0)
    def fn():
        calls.append(1)
        return "ok"

    result = fn()
    assert result == "ok"
    assert len(calls) == 1


def test_retry_raises_after_max_attempts():
    @retry(max_attempts=3, base_delay=0.0)
    def always_fail():
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        always_fail()


def test_retry_counts_attempts():
    calls = []

    @retry(max_attempts=3, base_delay=0.0)
    def fail_twice():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "done"

    result = fail_twice()
    assert result == "done"
    assert len(calls) == 3


def test_retry_does_not_catch_non_matching_exception():
    @retry(max_attempts=3, base_delay=0.0, exceptions=(OSError,))
    def raises_value_error():
        raise ValueError("not retried")

    with pytest.raises(ValueError):
        raises_value_error()


def test_retry_catches_matching_exception_and_retries():
    calls = []

    @retry(max_attempts=2, base_delay=0.0, exceptions=(OSError,))
    def os_then_success():
        calls.append(1)
        if len(calls) < 2:
            raise OSError("transient")
        return "ok"

    result = os_then_success()
    assert result == "ok"


def test_retry_preserves_function_name():
    @retry(max_attempts=2, base_delay=0.0)
    def my_function():
        return 42

    assert my_function.__name__ == "my_function"


def test_retry_returns_correct_value():
    @retry(max_attempts=1, base_delay=0.0)
    def returns_dict():
        return {"key": "value"}

    assert returns_dict() == {"key": "value"}


def test_retry_backoff_increases_delay():
    delays = []

    with patch("time.sleep", side_effect=lambda d: delays.append(d)):
        @retry(max_attempts=3, base_delay=1.0, backoff=2.0)
        def always_fail():
            raise ValueError("x")

        with pytest.raises(ValueError):
            always_fail()

    assert len(delays) == 2
    assert delays[1] > delays[0]


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_retry_max_attempts_param(max_attempts):
    calls = []

    @retry(max_attempts=max_attempts, base_delay=0.0)
    def always_fail():
        calls.append(1)
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        always_fail()

    assert len(calls) == max_attempts
