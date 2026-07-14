"""TTL cache tests."""

from __future__ import annotations

import time

import pytest

from app.cache import TTLCache


def test_set_and_get() -> None:
    c = TTLCache(ttl_seconds=10)
    c.set("k", 42)
    assert c.get("k") == 42


def test_missing_key() -> None:
    c = TTLCache(ttl_seconds=10)
    assert c.get("nonexistent") is None


def test_ttl_expiry() -> None:
    c = TTLCache(ttl_seconds=0)
    c.set("k", "v")
    time.sleep(0.01)
    assert c.get("k") is None


def test_invalidate() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", 1)
    c.invalidate("k")
    assert c.get("k") is None


def test_clear() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.size == 0
    assert c.hits == 0
    assert c.misses == 0


def test_max_size_eviction() -> None:
    c = TTLCache(ttl_seconds=60, max_size=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)
    assert c.size == 3


def test_hit_rate() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("x", 42)
    c.get("x")
    c.get("x")
    c.get("missing")
    assert c.hits == 2
    assert c.misses == 1
    assert c.hit_rate == pytest.approx(2 / 3, rel=0.01)


def test_evict_expired() -> None:
    c = TTLCache(ttl_seconds=0)
    c.set("x", 1)
    c.set("y", 2)
    time.sleep(0.01)
    removed = c.evict_expired()
    assert removed == 2
    assert c.size == 0


@pytest.mark.parametrize("value", [None, 0, "", [], {"x": 1}])
def test_stores_falsy_values(value: object) -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", value)
    assert c.get("k") == value


@pytest.mark.parametrize("ttl,size", [(5, 10), (60, 100), (300, 1000)])
def test_construction_params(ttl: int, size: int) -> None:
    c = TTLCache(ttl_seconds=ttl, max_size=size)
    assert c.size == 0


def test_get_or_set_returns_existing_value() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", "existing")
    assert c.get_or_set("k", "default") == "existing"


def test_get_or_set_stores_default_when_absent() -> None:
    c = TTLCache(ttl_seconds=60)
    result = c.get_or_set("new_key", "default_val")
    assert result == "default_val"
    assert c.get("new_key") == "default_val"


def test_stats_returns_expected_keys() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", "v")
    c.get("k")
    s = c.stats()
    assert set(s.keys()) == {"size", "hits", "misses", "hit_rate"}


def test_stats_values_consistent() -> None:
    c = TTLCache(ttl_seconds=60)
    c.set("k", "v")
    c.get("k")
    c.get("missing")
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["size"] == 1
    assert s["hit_rate"] == pytest.approx(0.5)
