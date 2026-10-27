"""Extended tests for Cyber-Guard app/rate_limit.py."""

from __future__ import annotations

import time
from collections import deque

import pytest

from app.rate_limit import WINDOW_SECONDS, _HITS, _prune, reset_rate_limit_state


class TestPrune:
    def test_removes_old_timestamps(self) -> None:
        bucket: deque = deque()
        now = time.monotonic()
        bucket.append(now - WINDOW_SECONDS - 1)
        bucket.append(now)
        _prune(bucket, now)
        assert len(bucket) == 1

    def test_keeps_recent_timestamps(self) -> None:
        bucket: deque = deque()
        now = time.monotonic()
        bucket.append(now - 5)
        bucket.append(now - 1)
        _prune(bucket, now)
        assert len(bucket) == 2

    def test_empty_bucket_no_error(self) -> None:
        bucket: deque = deque()
        _prune(bucket, time.monotonic())
        assert len(bucket) == 0

    def test_removes_all_stale(self) -> None:
        bucket: deque = deque()
        now = time.monotonic()
        for i in range(5):
            bucket.append(now - WINDOW_SECONDS - i)
        _prune(bucket, now)
        assert len(bucket) == 0


class TestResetRateLimitState:
    def test_clears_buckets(self) -> None:
        _HITS["test_key"].append(time.monotonic())
        reset_rate_limit_state()
        assert len(_HITS) == 0

    def test_idempotent_on_empty(self) -> None:
        reset_rate_limit_state()
        reset_rate_limit_state()
        assert len(_HITS) == 0


class TestWindowSeconds:
    def test_window_is_positive(self) -> None:
        assert WINDOW_SECONDS > 0

    def test_window_is_int(self) -> None:
        assert isinstance(WINDOW_SECONDS, int)
