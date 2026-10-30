"""Tests for PredictionCache."""
import pytest
from app.cache import PredictionCache


def test_set_and_get():
    c = PredictionCache()
    c.set("k1", {"price": 100})
    assert c.get("k1") == {"price": 100}


def test_miss_returns_none():
    c = PredictionCache()
    assert c.get("missing") is None


def test_lru_eviction():
    c = PredictionCache(max_size=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("a") is None
    assert c.get("b") == 2


def test_len():
    c = PredictionCache(max_size=10)
    c.set("x", 1)
    c.set("y", 2)
    assert len(c) == 2


def test_clear():
    c = PredictionCache()
    c.set("a", 1)
    c.clear()
    assert len(c) == 0


def test_contains():
    c = PredictionCache()
    c.set("z", 99)
    assert c.contains("z") is True
    assert c.contains("w") is False
