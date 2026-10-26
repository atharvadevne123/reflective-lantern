"""TTL/LRU cache tests for Ticket-Oracle."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


@pytest.fixture()
def cache():
    return TTLCache(maxsize=5, ttl=60.0)


def test_set_and_get(cache):
    cache.set("k1", {"priority": "P2"})
    assert cache.get("k1") == {"priority": "P2"}


def test_get_missing_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_expired_entry_returns_none():
    c = TTLCache(maxsize=10, ttl=0.05)
    c.set("k", "v")
    time.sleep(0.1)
    assert c.get("k") is None


def test_lru_eviction():
    c = TTLCache(maxsize=3, ttl=60.0)
    for i in range(4):
        c.set(f"k{i}", i)
    assert c.size <= 3
    assert c.get("k0") is None


def test_invalidate_removes_entry(cache):
    cache.set("k1", "v1")
    cache.invalidate("k1")
    assert cache.get("k1") is None


def test_invalidate_missing_is_noop(cache):
    cache.invalidate("does_not_exist")


def test_clear_empties_cache(cache):
    cache.set("k1", 1)
    cache.set("k2", 2)
    cache.clear()
    assert cache.size == 0


def test_size_reflects_entries(cache):
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.size == 2


def test_update_existing_key(cache):
    cache.set("k", "old")
    cache.set("k", "new")
    assert cache.get("k") == "new"
    assert cache.size == 1
