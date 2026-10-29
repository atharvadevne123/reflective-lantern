"""Extended tests for truncate_string and retry_with_backoff in utils."""

from __future__ import annotations

import pytest

from app.utils import retry_with_backoff, truncate_string


class TestTruncateString:
    def test_short_string_unchanged(self) -> None:
        assert truncate_string("hello", 80) == "hello"

    def test_exact_length_unchanged(self) -> None:
        s = "a" * 10
        assert truncate_string(s, 10) == s

    def test_long_string_truncated(self) -> None:
        s = "a" * 100
        result = truncate_string(s, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_custom_suffix(self) -> None:
        result = truncate_string("hello world", 8, suffix="--")
        assert result.endswith("--")
        assert len(result) == 8

    def test_empty_string(self) -> None:
        assert truncate_string("", 10) == ""

    def test_max_len_smaller_than_suffix(self) -> None:
        result = truncate_string("abcdef", 2, suffix="...")
        assert len(result) == 2


class TestRetryWithBackoff:
    def test_success_on_first_try(self) -> None:
        calls = []

        def fn():
            calls.append(1)
            return 42

        result = retry_with_backoff(fn, retries=3, base_delay=0.0)
        assert result == 42
        assert len(calls) == 1

    def test_retries_on_failure(self) -> None:
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        result = retry_with_backoff(fn, retries=5, base_delay=0.0)
        assert result == "ok"
        assert len(calls) == 3

    def test_raises_after_exhausting_retries(self) -> None:
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            retry_with_backoff(fn, retries=3, base_delay=0.0)

    def test_specific_exception_filter(self) -> None:
        calls = []

        def fn():
            calls.append(1)
            raise TypeError("type error")

        with pytest.raises(TypeError):
            retry_with_backoff(fn, retries=3, base_delay=0.0, exceptions=(TypeError,))
        assert len(calls) == 3

    def test_non_matching_exception_propagates_immediately(self) -> None:
        calls = []

        def fn():
            calls.append(1)
            raise KeyError("not retried")

        with pytest.raises(KeyError):
            retry_with_backoff(fn, retries=3, base_delay=0.0, exceptions=(ValueError,))
        assert len(calls) == 1
