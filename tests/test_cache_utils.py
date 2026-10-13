"""Tests for app.cache_utils module."""

from __future__ import annotations

import time

import pytest

from app.cache_utils import TTLCache, memoize_ttl


class TestTTLCacheSet:
    def test_set_and_get_returns_value(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_missing_key_returns_default(self) -> None:
        cache = TTLCache(ttl=60)
        assert cache.get("missing") is None

    def test_invalid_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="ttl"):
            TTLCache(ttl=0)

    def test_negative_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            TTLCache(ttl=60, max_size=-1)


class TestTTLCacheExpiry:
    def test_expired_entry_returns_default(self) -> None:
        cache = TTLCache(ttl=0.01)
        cache.set("k", "v")
        time.sleep(0.05)
        assert cache.get("k") is None

    def test_evict_expired_removes_stale(self) -> None:
        cache = TTLCache(ttl=0.01)
        cache.set("a", 1)
        time.sleep(0.05)
        removed = cache.evict_expired()
        assert removed >= 1


class TestMemoizeTtl:
    def test_caches_result(self) -> None:
        call_count = {"n": 0}

        @memoize_ttl(ttl=60)
        def expensive(x: int) -> int:
            call_count["n"] += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count["n"] == 1

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_parametrized_cache_hits(self, n: int) -> None:
        call_count = {"c": 0}

        @memoize_ttl(ttl=60)
        def f(x: int) -> int:
            call_count["c"] += 1
            return x

        for _ in range(n):
            f(42)
        assert call_count["c"] == 1
