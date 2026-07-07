"""Tests for the prediction cache module."""
from __future__ import annotations

import pytest


def test_cached_predict_miss() -> None:
    from app.cache import cached_predict, clear_cache

    clear_cache()
    result = cached_predict([0.0] * 25)
    assert result is None


def test_cache_and_retrieve() -> None:
    from app.cache import cache_prediction, cached_predict, clear_cache

    clear_cache()
    features = [1.0] * 25
    prediction = {"label": "attack", "confidence": 0.9}
    cache_prediction(features, prediction)
    result = cached_predict(features)
    assert result == prediction


def test_different_features_different_keys() -> None:
    from app.cache import cache_prediction, cached_predict, clear_cache

    clear_cache()
    cache_prediction([0.0] * 25, {"label": "normal"})
    assert cached_predict([1.0] * 25) is None


def test_cache_stats() -> None:
    from app.cache import cache_prediction, cache_stats, clear_cache

    clear_cache()
    assert cache_stats()["size"] == 0
    cache_prediction([0.0] * 25, {"label": "normal"})
    assert cache_stats()["size"] == 1


def test_clear_cache() -> None:
    from app.cache import cache_prediction, cached_predict, clear_cache

    cache_prediction([2.0] * 25, {"label": "attack"})
    clear_cache()
    assert cached_predict([2.0] * 25) is None


@pytest.mark.parametrize("n", [1, 5, 10])
def test_cache_multiple_entries(n: int) -> None:
    from app.cache import cache_prediction, cache_stats, clear_cache

    clear_cache()
    for i in range(n):
        cache_prediction([float(i)] * 25, {"label": "normal", "idx": i})
    assert cache_stats()["size"] == n
