"""Extended tests for TTLCache stats() and edge-case behaviours."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


class TestTTLCacheStats:
    def test_stats_returns_dict(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        result = cache.stats()
        assert isinstance(result, dict)

    def test_stats_has_size_key(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("a", 1)
        result = cache.stats()
        assert "size" in result

    def test_stats_has_maxsize_key(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        result = cache.stats()
        assert result["maxsize"] == 10

    def test_stats_size_reflects_entries(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("x", 1)
        cache.set("y", 2)
        assert cache.stats()["size"] == 2

    def test_stats_size_zero_after_clear(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("z", 99)
        cache.clear()
        assert cache.stats()["size"] == 0

    def test_maxsize_in_stats_matches_constructor(self) -> None:
        cache = TTLCache(maxsize=128)
        assert cache.stats()["maxsize"] == 128


class TestTTLCacheEdgeCases:
    def test_set_same_key_twice_keeps_one_entry(self) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("dup", "first")
        cache.set("dup", "second")
        assert len(cache) == 1
        assert cache.get("dup") == "second"

    def test_invalidate_nonexistent_key_no_error(self) -> None:
        cache = TTLCache(maxsize=5, ttl_seconds=60)
        cache.invalidate("nonexistent")

    def test_get_missing_key_returns_none(self) -> None:
        cache = TTLCache(maxsize=5, ttl_seconds=60)
        assert cache.get("missing") is None

    def test_eviction_when_full(self) -> None:
        cache = TTLCache(maxsize=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)
        assert len(cache) == 3
        assert cache.get("a") is None  # oldest evicted

    def test_expired_entry_not_returned(self) -> None:
        cache = TTLCache(maxsize=5, ttl_seconds=0.05)
        cache.set("exp", "val")
        time.sleep(0.1)
        assert cache.get("exp") is None

    @pytest.mark.parametrize("value", [None, 0, False, "", [], {}])
    def test_falsy_values_stored_correctly(self, value) -> None:
        cache = TTLCache(maxsize=10, ttl_seconds=60)
        cache.set("key", value)
        result = cache.get("key")
        assert result == value
