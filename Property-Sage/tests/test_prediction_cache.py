"""Tests for PredictionCache."""

import time

import pytest

from app.cache import PredictionCache


@pytest.fixture
def cache():
    return PredictionCache(ttl_seconds=60)


def test_set_and_get(cache):
    cache.set("k1", {"price": 100})
    assert cache.get("k1") == {"price": 100}


def test_miss_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_expired_entry():
    short_cache = PredictionCache(ttl_seconds=0)
    short_cache.set("k", "v")
    time.sleep(0.01)
    assert short_cache.get("k") is None


def test_make_key_deterministic():
    k1 = PredictionCache.make_key({"a": 1, "b": 2})
    k2 = PredictionCache.make_key({"b": 2, "a": 1})
    assert k1 == k2


def test_clear(cache):
    cache.set("x", 1)
    cache.clear()
    assert cache.size() == 0


@pytest.mark.parametrize("payload", [
    {"bedrooms": 2}, {"bedrooms": 3}, {"sqft": 1000}
])
def test_make_key_different_payloads(payload):
    k = PredictionCache.make_key(payload)
    assert isinstance(k, str)
    assert len(k) == 16
