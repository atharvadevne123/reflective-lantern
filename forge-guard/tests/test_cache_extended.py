"""Extended tests for forge-guard app/cache.py TTLCache."""

from __future__ import annotations

import time

from app.cache import TTLCache


class TestTTLCacheGet:
    def test_returns_none_on_miss(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        assert cache.get({"x": 1.0}) is None

    def test_returns_value_on_hit(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        data = {"a": 1.0, "b": 2.0}
        cache.set(data, "result")
        assert cache.get(data) == "result"

    def test_expired_entry_returns_none(self) -> None:
        cache = TTLCache(ttl_seconds=1)
        data = {"x": 0.5}
        cache.set(data, 42)
        time.sleep(1.1)
        assert cache.get(data) is None

    def test_key_order_independent(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        cache.set({"a": 1.0, "b": 2.0}, "value")
        assert cache.get({"b": 2.0, "a": 1.0}) == "value"


class TestTTLCacheSize:
    def test_size_zero_on_empty(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        assert cache.size() == 0

    def test_size_increases_on_set(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        cache.set({"x": 1.0}, "v1")
        cache.set({"y": 2.0}, "v2")
        assert cache.size() == 2

    def test_size_stable_on_overwrite(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        cache.set({"x": 1.0}, "v1")
        cache.set({"x": 1.0}, "v2")
        assert cache.size() == 1

    def test_clear_resets_size(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        cache.set({"x": 1.0}, "v")
        cache.clear()
        assert cache.size() == 0


class TestTTLCachePurge:
    def test_purge_removes_expired(self) -> None:
        cache = TTLCache(ttl_seconds=1)
        cache.set({"x": 1.0}, "v")
        time.sleep(1.1)
        removed = cache.purge()
        assert removed >= 1
        assert cache.size() == 0

    def test_purge_keeps_fresh(self) -> None:
        cache = TTLCache(ttl_seconds=60)
        cache.set({"x": 1.0}, "v")
        removed = cache.purge()
        assert removed == 0
        assert cache.size() == 1

    def test_purge_empty_cache(self) -> None:
        cache = TTLCache(ttl_seconds=10)
        assert cache.purge() == 0
