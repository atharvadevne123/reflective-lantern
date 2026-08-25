"""Tests for app.circuit_breaker module."""

from __future__ import annotations

import pytest

from app.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def _always_fail():
    raise ValueError("boom")


def _always_succeed():
    return "ok"


class TestCircuitBreakerClosed:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    def test_successful_call_returns_value(self):
        cb = CircuitBreaker()
        assert cb.call(_always_succeed) == "ok"

    def test_failure_increments_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        assert cb._failure_count == 1

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(_always_fail)
        assert cb.state is CircuitState.CLOSED


class TestCircuitBreakerOpens:
    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, expected_exceptions=(ValueError,))
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(_always_fail)
        assert cb.state is CircuitState.OPEN

    def test_open_circuit_raises_circuit_open_error(self):
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        with pytest.raises(CircuitOpenError):
            cb.call(_always_succeed)

    def test_open_circuit_does_not_call_function(self):
        calls = [0]

        def counting():
            calls[0] += 1

        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        with pytest.raises(CircuitOpenError):
            cb.call(counting)
        assert calls[0] == 0


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open_after_timeout(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        assert cb._state is CircuitState.OPEN
        monkeypatch.setattr("time.monotonic", lambda: cb._opened_at + 11)
        assert cb.state is CircuitState.HALF_OPEN

    def test_successful_probe_closes_circuit(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        monkeypatch.setattr("time.monotonic", lambda: cb._opened_at + 11)
        cb.call(_always_succeed)
        assert cb.state is CircuitState.CLOSED

    def test_failed_probe_reopens_circuit(self, monkeypatch):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        opened_at = cb._opened_at
        monkeypatch.setattr("time.monotonic", lambda: opened_at + 11)
        with pytest.raises(ValueError):
            cb.call(_always_fail)
        assert cb._state is CircuitState.OPEN


class TestCircuitBreakerDecorator:
    def test_decorator_usage(self):
        cb = CircuitBreaker(failure_threshold=2)

        @cb
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_decorator_preserves_name(self):
        cb = CircuitBreaker()

        @cb
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    @pytest.mark.parametrize("threshold", [1, 3, 5])
    def test_opens_at_various_thresholds(self, threshold):
        cb = CircuitBreaker(failure_threshold=threshold)
        for _ in range(threshold):
            with pytest.raises(ValueError):
                cb.call(_always_fail)
        assert cb._state is CircuitState.OPEN
