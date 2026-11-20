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


@pytest.mark.parametrize(
    "fail_times,max_attempts,expected",
    [
        (0, 3, "ok"),
        (1, 3, "ok"),
        (2, 3, "ok"),
    ],
)
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


class TestBackoffTiming:
    """Verify the delay schedule without paying real wall-clock time."""

    @staticmethod
    def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        recorded: list[float] = []
        monkeypatch.setattr("app.retry.time.sleep", recorded.append)
        return recorded

    def test_delay_grows_between_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps = self._record_sleeps(monkeypatch)

        @retry(exceptions=(_Boom,), max_attempts=4, base_delay=1.0, backoff=2.0, jitter=0.0)
        def always_fails() -> None:
            raise _Boom("down")

        with pytest.raises(_Boom):
            always_fails()
        assert sleeps == sorted(sleeps)
        assert len(sleeps) == 3

    def test_no_sleep_after_final_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps = self._record_sleeps(monkeypatch)

        @retry(exceptions=(_Boom,), max_attempts=3, base_delay=1.0, jitter=0.0)
        def always_fails() -> None:
            raise _Boom("down")

        with pytest.raises(_Boom):
            always_fails()
        # Three attempts means only two waits.
        assert len(sleeps) == 2

    def test_delay_is_capped_by_max_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps = self._record_sleeps(monkeypatch)

        @retry(exceptions=(_Boom,), max_attempts=6, base_delay=1.0, max_delay=3.0, backoff=10.0, jitter=0.0)
        def always_fails() -> None:
            raise _Boom("down")

        with pytest.raises(_Boom):
            always_fails()
        assert all(delay <= 3.0 for delay in sleeps)

    def test_no_sleep_when_first_attempt_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps = self._record_sleeps(monkeypatch)

        @retry(exceptions=(_Boom,), max_attempts=3)
        def succeeds() -> str:
            return "ok"

        assert succeeds() == "ok"
        assert sleeps == []


class TestRetrySemantics:
    def test_recovers_before_exhausting_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry(exceptions=(_Boom,), max_attempts=5)
        def flaky() -> str:
            calls[0] += 1
            if calls[0] < 3:
                raise _Boom("not yet")
            return "recovered"

        assert flaky() == "recovered"
        assert calls[0] == 3

    def test_unlisted_exception_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry(exceptions=(_Boom,), max_attempts=5)
        def wrong_error() -> None:
            calls[0] += 1
            raise _Other("different")

        with pytest.raises(_Other):
            wrong_error()
        assert calls[0] == 1

    def test_final_exception_is_the_last_one_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry(exceptions=(_Boom,), max_attempts=3)
        def numbered() -> None:
            calls[0] += 1
            raise _Boom(f"attempt {calls[0]}")

        with pytest.raises(_Boom, match="attempt 3"):
            numbered()

    def test_arguments_forwarded_on_every_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        seen: list[tuple] = []

        @retry(exceptions=(_Boom,), max_attempts=3)
        def takes_args(a: int, b: int = 0) -> int:
            seen.append((a, b))
            if len(seen) < 2:
                raise _Boom("retry")
            return a + b

        assert takes_args(2, b=3) == 5
        assert seen == [(2, 3), (2, 3)]

    def test_preserves_function_metadata(self) -> None:
        @retry(exceptions=(_Boom,))
        def documented() -> None:
            """Keep this docstring."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Keep this docstring."

    def test_single_attempt_never_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry(exceptions=(_Boom,), max_attempts=1)
        def once() -> None:
            calls[0] += 1
            raise _Boom("only once")

        with pytest.raises(_Boom):
            once()
        assert calls[0] == 1


class TestRetryOnNetworkError:
    def test_retries_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry_on_network_error(max_attempts=3)
        def flaky() -> str:
            calls[0] += 1
            if calls[0] < 3:
                raise ConnectionError("refused")
            return "connected"

        assert flaky() == "connected"

    def test_retries_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry_on_network_error(max_attempts=3)
        def slow() -> str:
            calls[0] += 1
            if calls[0] < 2:
                raise TimeoutError("timed out")
            return "done"

        assert slow() == "done"
        assert calls[0] == 2

    def test_does_not_retry_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.retry.time.sleep", lambda _: None)
        calls = [0]

        @retry_on_network_error(max_attempts=3)
        def bad_input() -> None:
            calls[0] += 1
            raise ValueError("not a network problem")

        with pytest.raises(ValueError):
            bad_input()
        assert calls[0] == 1
