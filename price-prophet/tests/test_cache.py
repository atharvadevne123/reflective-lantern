"""Tests for app/utils/cache.py."""
from __future__ import annotations

import time


def test_cache_set_get():
    from app.utils.cache import TTLCache
    c = TTLCache(ttl_seconds=60)
    c.set("k", 42)
    assert c.get("k") == 42


def test_cache_miss_returns_none():
    from app.utils.cache import TTLCache
    c = TTLCache(ttl_seconds=60)
    assert c.get("missing") is None


def test_cache_delete():
    from app.utils.cache import TTLCache
    c = TTLCache(ttl_seconds=60)
    c.set("k", 1)
    c.delete("k")
    assert c.get("k") is None


def test_cache_clear():
    from app.utils.cache import TTLCache
    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert len(c) == 0


def test_cache_expired():
    from app.utils.cache import TTLCache
    c = TTLCache(ttl_seconds=0)
    c.set("k", 99)
    time.sleep(0.01)
    assert c.get("k") is None
