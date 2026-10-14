"""Tests for the thread-safe TTL/LRU prediction cache."""
from __future__ import annotations

import time
import threading

import pytest

from app.cache import TTLCache


def test_set_and_get():
    cache = TTLCache(maxsize=10, ttl_seconds=60)
    cache.set("k1", {"signal": "BUY"})
    assert cache.get("k1") == {"signal": "BUY"}


def test_miss_returns_none():
    cache = TTLCache(maxsize=10, ttl_seconds=60)
    assert cache.get("missing") is None


def test_ttl_expiry():
    cache = TTLCache(maxsize=10, ttl_seconds=0.05)
    cache.set("k1", "value")
    time.sleep(0.1)
    assert cache.get("k1") is None


def test_lru_eviction():
    cache = TTLCache(maxsize=3, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    # Access 'a' to make it recently used
    cache.get("a")
    # 'b' is now LRU; adding 'd' should evict 'b'
    cache.set("d", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("d") == 4


def test_overwrite_existing_key():
    cache = TTLCache(maxsize=10, ttl_seconds=60)
    cache.set("k", "old")
    cache.set("k", "new")
    assert cache.get("k") == "new"


def test_clear():
    cache = TTLCache(maxsize=10, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_size_zero_maxsize_allows_no_entries():
    cache = TTLCache(maxsize=0, ttl_seconds=60)
    cache.set("k", 1)
    assert cache.get("k") is None


def test_thread_safety():
    cache = TTLCache(maxsize=500, ttl_seconds=60)
    errors = []

    def writer(prefix: str):
        try:
            for i in range(50):
                cache.set(f"{prefix}-{i}", i)
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for i in range(50):
                cache.get(f"a-{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=("a",)) for _ in range(5)]
    threads += [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
