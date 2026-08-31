"""Tests for app.circuit_breaker module."""

from __future__ import annotations

import time

import pytest

from app.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


def _always_fail():
    raise ValueError("boom")


def _always_succeed():
    return "ok"


def _always_fails() -> None:
    """Raise a RuntimeError, for tests using the default expected_exceptions."""
    raise RuntimeError("boom")


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


class TestStateTransitions:
    """Cover the CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle."""

    def test_stays_closed_below_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(_always_fails)
        assert breaker.state is CircuitState.CLOSED

    def test_opens_exactly_at_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(_always_fails)
        assert breaker.state is CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(_always_fails)
        breaker.call(lambda: "ok")
        # Count was reset, so two more failures must not open the circuit.
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(_always_fails)
        assert breaker.state is CircuitState.CLOSED

    def test_open_circuit_blocks_without_calling(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        with pytest.raises(RuntimeError):
            breaker.call(_always_fails)

        calls: list[int] = []

        def _tracked() -> str:
            calls.append(1)
            return "ok"

        with pytest.raises(CircuitOpenError):
            breaker.call(_tracked)
        assert calls == []

    def test_half_open_after_recovery_timeout(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        with pytest.raises(RuntimeError):
            breaker.call(_always_fails)
        assert breaker.state is CircuitState.OPEN
        time.sleep(0.02)
        assert breaker.state is CircuitState.HALF_OPEN

    def test_successful_probe_closes_circuit(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        with pytest.raises(RuntimeError):
            breaker.call(_always_fails)
        time.sleep(0.02)
        assert breaker.call(lambda: "recovered") == "recovered"
        assert breaker.state is CircuitState.CLOSED

    def test_failed_probe_reopens_immediately(self) -> None:
        # In HALF_OPEN a single failure re-opens, regardless of threshold.
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=0.01)
        for _ in range(5):
            with pytest.raises(RuntimeError):
                breaker.call(_always_fails)
        time.sleep(0.02)
        assert breaker.state is CircuitState.HALF_OPEN
        with pytest.raises(RuntimeError):
            breaker.call(_always_fails)
        assert breaker.state is CircuitState.OPEN


class TestExpectedExceptions:
    def test_unexpected_exception_propagates_without_tripping(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, expected_exceptions=(ValueError,))

        def _raises_type_error() -> None:
            raise TypeError("not counted")

        with pytest.raises(TypeError):
            breaker.call(_raises_type_error)
        assert breaker.state is CircuitState.CLOSED

    def test_expected_exception_trips_circuit(self) -> None:
        breaker = CircuitBreaker(failure_threshold=1, expected_exceptions=(ValueError,))

        def _raises_value_error() -> None:
            raise ValueError("counted")

        with pytest.raises(ValueError):
            breaker.call(_raises_value_error)
        assert breaker.state is CircuitState.OPEN

    def test_multiple_expected_types_all_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, expected_exceptions=(ValueError, KeyError))

        def _raises_value() -> None:
            raise ValueError("first")

        def _raises_key() -> None:
            raise KeyError("second")

        with pytest.raises(ValueError):
            breaker.call(_raises_value)
        with pytest.raises(KeyError):
            breaker.call(_raises_key)
        assert breaker.state is CircuitState.OPEN


class TestDecoratorUsage:
    def test_decorator_forwards_arguments(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2)

        @breaker
        def add(a: int, b: int = 0) -> int:
            return a + b

        assert add(2, b=3) == 5

    def test_decorator_preserves_metadata(self) -> None:
        breaker = CircuitBreaker()

        @breaker
        def documented() -> None:
            """A docstring worth keeping."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "A docstring worth keeping."

    def test_decorator_trips_on_repeated_failure(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        @breaker
        def flaky() -> None:
            raise RuntimeError("down")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                flaky()
        with pytest.raises(CircuitOpenError):
            flaky()

    def test_return_value_passes_through(self) -> None:
        breaker = CircuitBreaker()
        assert breaker.call(lambda x: x * 2, 21) == 42
