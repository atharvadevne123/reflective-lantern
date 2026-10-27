"""Tests for veritas-rag app/caching.py."""

from __future__ import annotations

import time

from app.caching import RetrievalCache, cache_key, normalize_query


class TestNormalizeQuery:
    def test_lowercases(self) -> None:
        assert normalize_query("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self) -> None:
        result = normalize_query("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_single_spaces(self) -> None:
        result = normalize_query("hello   world")
        assert "  " not in result

    def test_empty_string(self) -> None:
        assert normalize_query("") == ""

    def test_preserves_words(self) -> None:
        result = normalize_query("what is the capital")
        assert "what" in result
        assert "capital" in result


class TestCacheKey:
    def test_returns_string(self) -> None:
        key = cache_key("test query")
        assert isinstance(key, str)

    def test_fixed_length(self) -> None:
        key = cache_key("any query")
        assert len(key) == 32

    def test_deterministic(self) -> None:
        assert cache_key("same") == cache_key("same")

    def test_different_queries_different_keys(self) -> None:
        assert cache_key("query a") != cache_key("query b")


class TestRetrievalCache:
    def test_miss_on_empty(self) -> None:
        cache = RetrievalCache()
        assert cache.get("anything") is None

    def test_set_and_get(self) -> None:
        cache = RetrievalCache()
        result = {"chunks": ["a"]}
        cache.set("query", result)
        assert cache.get("query") == result

    def test_hit_increments_counter(self) -> None:
        cache = RetrievalCache()
        cache.set("q", {"data": 1})
        before_hits = cache._counters.get("hits", 0)
        cache.get("q")
        assert cache._counters.get("hits", 0) > before_hits

    def test_miss_increments_counter(self) -> None:
        cache = RetrievalCache()
        before_misses = cache._counters.get("misses", 0)
        cache.get("not_cached")
        assert cache._counters.get("misses", 0) > before_misses

    def test_evicts_oldest_when_full(self) -> None:
        cache = RetrievalCache(max_size=3)
        for i in range(4):
            cache.set(f"q{i}", {"i": i})
        assert len(cache._cache) <= 3

    def test_ttl_expiry(self) -> None:
        cache = RetrievalCache(ttl_seconds=0.05)
        cache.set("q", {"data": 1})
        time.sleep(0.1)
        assert cache.get("q") is None

    def test_stats_returns_dict(self) -> None:
        cache = RetrievalCache()
        stats = cache.stats()
        assert isinstance(stats, dict)
