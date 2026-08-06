"""Tests for the TTL prediction cache."""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def cache():
    from app.cache import TTLCache

    return TTLCache(ttl_seconds=1, max_size=5)


SAMPLE = {"temperature": 75.0, "pressure": 50.0, "vibration": 2.1,
          "cycle_time": 28.0, "tool_wear": 15.0, "power_consumption": 98.0,
          "humidity": 45.0}


def test_cache_miss_returns_none(cache):
    assert cache.get(SAMPLE) is None


def test_cache_hit_returns_value(cache):
    cache.set(SAMPLE, (1, 0.9))
    result = cache.get(SAMPLE)
    assert result == (1, 0.9)


def test_cache_different_readings_stored_separately(cache):
    reading_a = {**SAMPLE, "temperature": 80.0}
    reading_b = {**SAMPLE, "temperature": 90.0}
    cache.set(reading_a, (0, 0.1))
    cache.set(reading_b, (1, 0.9))
    assert cache.get(reading_a) == (0, 0.1)
    assert cache.get(reading_b) == (1, 0.9)


def test_cache_expiry(cache):
    cache.set(SAMPLE, (1, 0.8))
    assert cache.get(SAMPLE) == (1, 0.8)
    time.sleep(1.1)
    assert cache.get(SAMPLE) is None


def test_cache_size(cache):
    assert cache.size() == 0
    cache.set(SAMPLE, (0, 0.2))
    assert cache.size() == 1


def test_cache_clear(cache):
    cache.set(SAMPLE, (0, 0.3))
    cache.clear()
    assert cache.size() == 0
    assert cache.get(SAMPLE) is None


def test_cache_purge_removes_expired(cache):
    cache.set(SAMPLE, (1, 0.7))
    time.sleep(1.1)
    removed = cache.purge()
    assert removed == 1
    assert cache.size() == 0


def test_cache_max_size_eviction():
    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=60, max_size=3)
    for i in range(4):
        reading = {**SAMPLE, "temperature": float(i * 10)}
        c.set(reading, i)
    assert c.size() <= 3


def test_cache_thread_safety():
    import threading

    from app.cache import TTLCache

    c = TTLCache(ttl_seconds=60, max_size=1000)
    errors = []

    def writer(idx: int) -> None:
        try:
            reading = {**SAMPLE, "temperature": float(idx)}
            c.set(reading, idx)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert c.size() <= 50


def test_cache_overwrite_same_key(cache):
    cache.set(SAMPLE, (0, 0.1))
    cache.set(SAMPLE, (1, 0.9))
    assert cache.get(SAMPLE) == (1, 0.9)


def test_cache_purge_returns_zero_when_none_expired(cache):
    cache.set(SAMPLE, (0, 0.5))
    removed = cache.purge()
    assert removed == 0
