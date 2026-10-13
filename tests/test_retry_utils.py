"""Tests for app.retry_utils module."""

from __future__ import annotations

import pytest

from app.retry_utils import RetryError, compute_backoff_seconds, retry, retry_with_backoff


class TestComputeBackoffSeconds:
    def test_first_attempt_returns_base(self) -> None:
        assert compute_backoff_seconds(1, base=1.0, multiplier=2.0) == pytest.approx(1.0)

    def test_second_attempt_doubles(self) -> None:
        assert compute_backoff_seconds(2, base=1.0, multiplier=2.0) == pytest.approx(2.0)

    def test_third_attempt_quadruples(self) -> None:
        assert compute_backoff_seconds(3, base=1.0, multiplier=2.0) == pytest.approx(4.0)

    def test_capped_at_max(self) -> None:
        assert compute_backoff_seconds(10, base=1.0, multiplier=2.0, cap=30.0) == pytest.approx(30.0)

    @pytest.mark.parametrize("attempt,expected", [(1, 0.5), (2, 1.0), (3, 2.0)])
    def test_parametrized_base_half(self, attempt: int, expected: float) -> None:
        assert compute_backoff_seconds(attempt, base=0.5, multiplier=2.0) == pytest.approx(expected)

    def test_cap_equals_base_always_returns_cap(self) -> None:
        for attempt in range(1, 6):
            assert compute_backoff_seconds(attempt, base=5.0, multiplier=2.0, cap=5.0) == pytest.approx(5.0)


class TestRetry:
    def test_succeeds_on_first_try(self) -> None:
        result = retry(lambda: 42, attempts=3)
        assert result == 42

    def test_retries_until_success(self) -> None:
        call_count = {"n": 0}

        def flaky() -> int:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return 99

        result = retry(flaky, attempts=3, exceptions=(ValueError,))
        assert result == 99

    def test_raises_retry_error_after_exhaustion(self) -> None:
        with pytest.raises(RetryError):
            retry(lambda: (_ for _ in ()).throw(ValueError("always")), attempts=3, exceptions=(ValueError,))

    def test_zero_attempts_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            retry(lambda: 1, attempts=0)

    def test_passes_args_to_func(self) -> None:
        result = retry(lambda x, y: x + y, 3, 4, attempts=1)
        assert result == 7

    def test_only_retries_specified_exception(self) -> None:
        def raises_type_error() -> None:
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            retry(raises_type_error, attempts=3, exceptions=(ValueError,))

    def test_single_attempt(self) -> None:
        result = retry(lambda: "ok", attempts=1)
        assert result == "ok"

    @pytest.mark.parametrize("attempts", [1, 2, 5])
    def test_attempt_count_parametrized(self, attempts: int) -> None:
        counter = {"n": 0}

        def always_fail() -> None:
            counter["n"] += 1
            raise RuntimeError("fail")

        with pytest.raises(RetryError):
            retry(always_fail, attempts=attempts, exceptions=(RuntimeError,))
        assert counter["n"] == attempts


class TestRetryWithBackoff:
    def test_succeeds_immediately(self) -> None:
        result = retry_with_backoff(lambda: "done", attempts=3, base=0.001)
        assert result == "done"

    def test_retries_on_exception(self) -> None:
        state = {"n": 0}

        def flaky() -> str:
            state["n"] += 1
            if state["n"] < 2:
                raise ConnectionError("retry me")
            return "ok"

        result = retry_with_backoff(flaky, attempts=3, base=0.001, exceptions=(ConnectionError,))
        assert result == "ok"

    def test_raises_retry_error_after_exhaustion(self) -> None:
        with pytest.raises(RetryError):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(OSError("nope")),
                attempts=2,
                base=0.001,
                exceptions=(OSError,),
            )

    def test_zero_attempts_raises(self) -> None:
        with pytest.raises(ValueError):
            retry_with_backoff(lambda: 1, attempts=0)

    def test_non_retried_exception_propagates(self) -> None:
        with pytest.raises(KeyError):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(KeyError("k")),
                attempts=3,
                base=0.001,
                exceptions=(ValueError,),
            )
