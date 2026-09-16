"""Retry and circuit breaker tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.retry import CircuitBreaker, retry


def test_retry_succeeds_first_attempt():
    calls = []

    @retry(max_attempts=3)
    def ok():
        calls.append(1)
        return "done"

    result = ok()
    assert result == "done"
    assert len(calls) == 1


def test_retry_retries_on_failure(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
    attempts = []

    @retry(max_attempts=3, exceptions=(ValueError,))
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("not yet")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert len(attempts) == 3


def test_retry_raises_after_max(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda _: None)

    @retry(max_attempts=2, exceptions=(RuntimeError,))
    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        always_fails()


def test_retry_ignores_unlisted_exception(monkeypatch):
    monkeypatch.setattr("app.retry.time.sleep", lambda _: None)

    @retry(max_attempts=3, exceptions=(ValueError,))
    def raises_type_error():
        raise TypeError("wrong type")

    with pytest.raises(TypeError):
        raises_type_error()


def test_circuit_breaker_open_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=999.0)

    def bad():
        raise RuntimeError("fail")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(bad)

    assert cb.is_open


def test_circuit_breaker_blocks_when_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999.0)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

    with pytest.raises(RuntimeError, match="Circuit breaker"):
        cb.call(lambda: None)


def test_circuit_breaker_resets_on_success():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert not cb.is_open
    assert cb._failures == 0


def test_circuit_breaker_passthrough_when_closed():
    cb = CircuitBreaker()
    result = cb.call(lambda: 42)
    assert result == 42
