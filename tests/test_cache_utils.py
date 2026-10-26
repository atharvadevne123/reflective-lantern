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

    def test_missing_key_returns_custom_default(self) -> None:
        cache = TTLCache(ttl=60)
        assert cache.get("missing", "fallback") == "fallback"

    def test_overwrite_updates_value(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2

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

    def test_live_entry_not_expired(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("k", "alive")
        assert cache.get("k") == "alive"

    def test_evict_expired_removes_stale(self) -> None:
        cache = TTLCache(ttl=0.01)
        cache.set("a", 1)
        cache.set("b", 2)
        time.sleep(0.05)
        removed = cache.evict_expired()
        assert removed == 2
        assert len(cache) == 0


class TestTTLCacheContainsAndLen:
    def test_contains_true_for_live_key(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("x", 10)
        assert "x" in cache

    def test_contains_false_for_missing_key(self) -> None:
        cache = TTLCache(ttl=60)
        assert "missing" not in cache

    def test_len_reflects_stored_count(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_delete_reduces_len(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("a", 1)
        cache.delete("a")
        assert len(cache) == 0

    def test_delete_nonexistent_returns_false(self) -> None:
        cache = TTLCache(ttl=60)
        assert cache.delete("ghost") is False


class TestTTLCacheMaxSize:
    def test_max_size_evicts_oldest(self) -> None:
        cache = TTLCache(ttl=60, max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert len(cache) == 2

    def test_max_size_zero_unlimited(self) -> None:
        cache = TTLCache(ttl=60, max_size=0)
        for i in range(100):
            cache.set(str(i), i)
        assert len(cache) == 100


class TestTTLCacheKeys:
    def test_keys_returns_non_expired(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("x", 1)
        cache.set("y", 2)
        assert "x" in cache
        assert "y" in cache

    def test_size_counts_live_entries(self) -> None:
        cache = TTLCache(ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size() == 2


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

    def test_different_args_call_func_again(self) -> None:
        call_count = {"n": 0}

        @memoize_ttl(ttl=60)
        def add(a: int, b: int) -> int:
            call_count["n"] += 1
            return a + b

        add(1, 2)
        add(3, 4)
        assert call_count["n"] == 2

    def test_expired_entry_recalculates(self) -> None:
        call_count = {"n": 0}

        @memoize_ttl(ttl=0.01)
        def func(x: int) -> int:
            call_count["n"] += 1
            return x

        func(1)
        time.sleep(0.05)
        func(1)
        assert call_count["n"] == 2

    def test_wrapped_attribute_preserved(self) -> None:
        @memoize_ttl(ttl=60)
        def identity(x: int) -> int:
            return x

        assert hasattr(identity, "__wrapped__")
        assert hasattr(identity, "__cache__")

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
