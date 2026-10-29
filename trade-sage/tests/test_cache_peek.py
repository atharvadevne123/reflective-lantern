"""Tests for TTLCache.peek, TTLCache.keys, and get_cache_key."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache, get_cache_key


class TestTTLCachePeek:
    def test_peek_returns_value(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("k", "v")
        assert cache.peek("k") == "v"

    def test_peek_missing_key_returns_none(self) -> None:
        cache = TTLCache()
        assert cache.peek("nope") is None

    def test_peek_does_not_update_lru(self) -> None:
        cache = TTLCache(maxsize=2, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.peek("a")
        cache.set("c", 3)
        assert cache.get("b") is None or cache.get("a") is None

    def test_peek_expired_returns_none(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=0.01)
        cache.set("x", 99)
        time.sleep(0.05)
        assert cache.peek("x") is None


class TestTTLCacheKeys:
    def test_returns_list(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        assert isinstance(cache.keys(), list)

    def test_contains_live_keys(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("x", 1)
        assert "x" in cache.keys()

    def test_excludes_expired_keys(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=0.01)
        cache.set("y", 2)
        time.sleep(0.05)
        assert "y" not in cache.keys()

    def test_empty_cache_returns_empty_list(self) -> None:
        cache = TTLCache()
        assert cache.keys() == []


class TestGetCacheKey:
    def test_single_part(self) -> None:
        assert get_cache_key("hello") == "hello"

    def test_multiple_parts_joined(self) -> None:
        assert get_cache_key("AAPL", 5, 0.95) == "AAPL|5|0.95"

    def test_consistent_output(self) -> None:
        assert get_cache_key("a", "b", "c") == get_cache_key("a", "b", "c")

    def test_different_inputs_differ(self) -> None:
        assert get_cache_key("a", "b") != get_cache_key("b", "a")

    def test_empty_parts(self) -> None:
        assert get_cache_key() == ""
