"""Cache tests for Risk-Sense."""

from __future__ import annotations

import pytest

from app.cache import (
    cache_prediction,
    cache_stats,
    cached_risk_thresholds,
    clear_cache,
    get_cached_prediction,
)


@pytest.fixture(autouse=True)
def reset_cache() -> None:
    clear_cache()


class TestPredictionCache:
    def test_miss_returns_none(self) -> None:
        features = {"debt_to_income": 0.3, "annual_income": 50000.0}
        assert get_cached_prediction(features) is None

    def test_hit_returns_cached_value(self) -> None:
        features = {"debt_to_income": 0.3, "annual_income": 50000.0}
        cache_prediction(features, 0.25)
        result = get_cached_prediction(features)
        assert result == pytest.approx(0.25)

    def test_different_features_different_keys(self) -> None:
        f1 = {"debt_to_income": 0.3}
        f2 = {"debt_to_income": 0.9}
        cache_prediction(f1, 0.1)
        cache_prediction(f2, 0.9)
        assert get_cached_prediction(f1) == pytest.approx(0.1)
        assert get_cached_prediction(f2) == pytest.approx(0.9)

    def test_clear_cache_removes_entries(self) -> None:
        features = {"debt_to_income": 0.3}
        cache_prediction(features, 0.5)
        clear_cache()
        assert get_cached_prediction(features) is None

    def test_cache_stats_size_increases(self) -> None:
        assert cache_stats()["size"] == 0
        cache_prediction({"key": "a"}, 0.1)
        assert cache_stats()["size"] == 1

    def test_cache_stats_has_max_size(self) -> None:
        stats = cache_stats()
        assert "max_size" in stats
        assert stats["max_size"] > 0


class TestCachedRiskThresholds:
    def test_default_thresholds(self) -> None:
        low, medium, high = cached_risk_thresholds()
        assert low == 0.2
        assert medium == 0.5
        assert high == 0.75

    def test_custom_thresholds(self) -> None:
        low, medium, high = cached_risk_thresholds(0.1, 0.4, 0.7)
        assert low == 0.1
        assert medium == 0.4
        assert high == 0.7
