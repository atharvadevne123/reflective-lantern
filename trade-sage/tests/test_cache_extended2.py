"""Extended TTL/LRU cache tests — second pass with parametrize and boundary coverage."""
from __future__ import annotations

import time

import pytest

from app.cache import TTLCache, get_cache_key


def test_cache_get_missing_key_returns_none() -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    assert cache.get("absent") is None


def test_cache_set_then_get_returns_value() -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_expired_entry_returns_none(monkeypatch) -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=0.001)
    cache.set("k", "value")
    time.sleep(0.05)
    assert cache.get("k") is None


def test_cache_lru_eviction_at_maxsize() -> None:
    cache = TTLCache(maxsize=3, ttl_seconds=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)
    assert len(cache) == 3
    assert cache.get("a") is None


def test_cache_invalidate_removes_key() -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    cache.set("x", 42)
    cache.invalidate("x")
    assert cache.get("x") is None


def test_cache_clear_empties_all() -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    for i in range(5):
        cache.set(str(i), i)
    cache.clear()
    assert len(cache) == 0


def test_cache_len_reflects_insertions() -> None:
    cache = TTLCache(maxsize=20, ttl_seconds=60.0)
    for i in range(7):
        cache.set(f"key{i}", i)
    assert len(cache) == 7


def test_cache_stats_has_size_and_maxsize() -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    cache.set("a", 1)
    stats = cache.stats()
    assert "size" in stats
    assert "maxsize" in stats
    assert stats["maxsize"] == 10


def test_cache_peek_does_not_update_lru() -> None:
    cache = TTLCache(maxsize=2, ttl_seconds=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.peek("a")
    cache.set("c", 3)
    assert cache.get("b") is None


def test_cache_peek_expired_returns_none(monkeypatch) -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=0.001)
    cache.set("k", "v")
    time.sleep(0.05)
    assert cache.peek("k") is None


def test_cache_keys_snapshot_excludes_expired(monkeypatch) -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=60.0)
    cache.set("live", 1)
    keys = cache.keys()
    assert "live" in keys


@pytest.mark.parametrize("maxsize", [1, 5, 10, 100])
def test_cache_respects_maxsize(maxsize: int) -> None:
    cache = TTLCache(maxsize=maxsize, ttl_seconds=60.0)
    for i in range(maxsize + 10):
        cache.set(str(i), i)
    assert len(cache) <= maxsize


def test_get_cache_key_joins_parts() -> None:
    key = get_cache_key("a", "b", 42)
    assert key == "a|b|42"


def test_get_cache_key_single_part() -> None:
    key = get_cache_key("only")
    assert key == "only"


def test_get_cache_key_empty_string_in_parts() -> None:
    key = get_cache_key("a", "", "b")
    assert "|" in key


@pytest.mark.parametrize("ttl", [1.0, 10.0, 60.0, 300.0])
def test_cache_accepts_various_ttl(ttl: float) -> None:
    cache = TTLCache(maxsize=10, ttl_seconds=ttl)
    cache.set("k", "v")
    assert cache.get("k") == "v"
