"""Tests for app.retry module."""

from __future__ import annotations

import pytest

from app.retry import retry, retry_on_network_error


class _Boom(Exception):
    pass


class _Other(Exception):
    pass


def _make_flaky(fail_times: int, exc: type = _Boom):
    """Return a function that raises exc for the first fail_times calls."""
    calls = [0]

    def fn():
        calls[0] += 1
        if calls[0] <= fail_times:
            raise exc(f"fail #{calls[0]}")
        return "ok"

    return fn


@pytest.mark.parametrize("fail_times,max_attempts,expected", [
    (0, 3, "ok"),
    (1, 3, "ok"),
    (2, 3, "ok"),
])
def test_retry_succeeds_after_failures(fail_times, max_attempts, expected, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    fn = _make_flaky(fail_times)
    wrapped = retry(exceptions=(_Boom,), max_attempts=max_attempts, base_delay=0)(fn)
    assert wrapped() == expected


def test_retry_raises_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    fn = _make_flaky(5)
    wrapped = retry(exceptions=(_Boom,), max_attempts=3, base_delay=0)(fn)
    with pytest.raises(_Boom):
        wrapped()


def test_retry_does_not_catch_unlisted_exceptions(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    fn = _make_flaky(1, exc=_Other)
    wrapped = retry(exceptions=(_Boom,), max_attempts=3, base_delay=0)(fn)
    with pytest.raises(_Other):
        wrapped()


def test_retry_preserves_function_name():
    def my_func():
        return 1

    wrapped = retry()(my_func)
    assert wrapped.__name__ == "my_func"


def test_retry_passes_args_and_kwargs(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = [0]

    @retry(exceptions=(_Boom,), max_attempts=2, base_delay=0)
    def add(a, b=0):
        calls[0] += 1
        if calls[0] == 1:
            raise _Boom()
        return a + b

    assert add(3, b=4) == 7


def test_retry_on_network_error_returns_callable():
    decorator = retry_on_network_error(max_attempts=2, base_delay=0)
    assert callable(decorator)


def test_retry_on_network_error_retries_on_connection_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = [0]

    @retry_on_network_error(max_attempts=3, base_delay=0)
    def flaky():
        calls[0] += 1
        if calls[0] < 3:
            raise ConnectionError("timeout")
        return "done"

    assert flaky() == "done"
    assert calls[0] == 3


@pytest.mark.parametrize("max_attempts", [1, 2, 5])
def test_retry_attempt_count_respected(max_attempts, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = [0]

    @retry(exceptions=(_Boom,), max_attempts=max_attempts, base_delay=0)
    def always_fail():
        calls[0] += 1
        raise _Boom()

    with pytest.raises(_Boom):
        always_fail()
    assert calls[0] == max_attempts
