"""Extended caching tests with parametrize and boundary coverage."""
from __future__ import annotations

import time

import pytest

from app.caching import (
    RetrievalCache,
    cache_age_seconds,
    cache_fill_ratio,
    cache_key,
    normalize_query,
    top_cached_queries,
)


def test_normalize_query_lowercase() -> None:
    assert normalize_query("HELLO WORLD") == "hello world"


def test_normalize_query_removes_punctuation() -> None:
    result = normalize_query("hello, world!")
    assert "," not in result
    assert "!" not in result


def test_normalize_query_collapses_whitespace() -> None:
    result = normalize_query("hello   world")
    assert result == "hello world"


@pytest.mark.parametrize("query,expected", [
    ("What is AI?", "what is ai"),
    ("  spaces  ", "spaces"),
    ("UPPER CASE", "upper case"),
    ("hello, world!", "hello world"),
])
def test_normalize_query_parametrized(query: str, expected: str) -> None:
    assert normalize_query(query) == expected


def test_cache_key_deterministic() -> None:
    key1 = cache_key("hello world")
    key2 = cache_key("hello world")
    assert key1 == key2


def test_cache_key_different_queries_different_keys() -> None:
    assert cache_key("query one") != cache_key("query two")


def test_cache_key_normalized_equivalent_queries_same_key() -> None:
    assert cache_key("Hello, World!") == cache_key("hello world")


def test_cache_key_length_32() -> None:
    assert len(cache_key("test query")) == 32


def test_retrieval_cache_hit() -> None:
    cache = RetrievalCache(ttl_seconds=60.0)
    cache.put("test query", {"result": "value"})
    result = cache.get("test query")
    assert result == {"result": "value"}


def test_retrieval_cache_miss_returns_none() -> None:
    cache = RetrievalCache()
    result = cache.get("not cached query")
    assert result is None


def test_retrieval_cache_miss_increments_counter() -> None:
    cache = RetrievalCache()
    cache.get("uncached query")
    assert cache.misses == 1


def test_retrieval_cache_hit_increments_counter() -> None:
    cache = RetrievalCache(ttl_seconds=60.0)
    cache.put("test", "value")
    cache.get("test")
    assert cache.hits == 1


def test_retrieval_cache_lru_eviction_at_max_size() -> None:
    cache = RetrievalCache(max_size=5, ttl_seconds=60.0)
    for i in range(6):
        cache.put(f"query {i}", f"value {i}")
    assert len(cache) <= 5


@pytest.mark.parametrize("max_size", [1, 10, 100])
def test_retrieval_cache_respects_max_size(max_size: int) -> None:
    cache = RetrievalCache(max_size=max_size, ttl_seconds=60.0)
    for i in range(max_size + 5):
        cache.put(f"query {i}", i)
    assert len(cache) <= max_size


def test_retrieval_cache_stats_has_required_keys() -> None:
    cache = RetrievalCache()
    stats = cache.stats()
    for key in ("size", "hits", "misses", "hit_rate"):
        assert key in stats


def test_retrieval_cache_hit_rate_zero_on_empty() -> None:
    cache = RetrievalCache()
    assert cache.stats()["hit_rate"] == 0.0


def test_cache_fill_ratio_empty_returns_zero() -> None:
    cache = RetrievalCache(max_size=100)
    assert cache_fill_ratio(cache) == 0.0


def test_cache_fill_ratio_full() -> None:
    cache = RetrievalCache(max_size=5, ttl_seconds=60.0)
    for i in range(5):
        cache.put(f"q{i}", i)
    ratio = cache_fill_ratio(cache)
    assert 0.9 <= ratio <= 1.0


def test_top_cached_queries_returns_at_most_n() -> None:
    cache = RetrievalCache(max_size=100, ttl_seconds=60.0)
    for i in range(10):
        cache.put(f"query {i}", i)
    result = top_cached_queries(cache, n=3)
    assert len(result) <= 3


def test_cache_age_seconds_missing_returns_none() -> None:
    cache = RetrievalCache()
    assert cache_age_seconds(cache, "not cached") is None


def test_cache_age_seconds_non_negative() -> None:
    cache = RetrievalCache(ttl_seconds=60.0)
    cache.put("test query", "value")
    age = cache_age_seconds(cache, "test query")
    assert age is not None
    assert age >= 0.0
