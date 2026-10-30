"""Tests for app.stats helpers."""

import pytest

from app.stats import neighborhood_benchmark, neighborhood_price_stats


def test_price_stats_basic():
    stats = neighborhood_price_stats([100.0, 200.0, 300.0])
    assert stats["mean"] == 200.0
    assert stats["count"] == 3


def test_price_stats_empty():
    stats = neighborhood_price_stats([])
    assert stats["count"] == 0
    assert stats["mean"] == 0.0


def test_price_stats_single():
    stats = neighborhood_price_stats([500.0])
    assert stats["std_dev"] == 0.0


def test_benchmark_above_median():
    result = neighborhood_benchmark(300.0, [100.0, 200.0, 250.0, 400.0])
    assert result["above_median"] is True


def test_benchmark_empty_neighborhood():
    result = neighborhood_benchmark(300.0, [])
    assert result["percentile_rank"] == 0.0


@pytest.mark.parametrize("subject,expected_ge", [
    (400.0, 60.0),
    (50.0, 0.0),
])
def test_benchmark_percentile(subject, expected_ge):
    prices = [100.0, 200.0, 300.0, 400.0, 500.0]
    result = neighborhood_benchmark(subject, prices)
    assert result["percentile_rank"] >= expected_ge


def test_investment_score_basic():
    from app.stats import property_investment_score
    score = property_investment_score(0.06, 200.0, 220.0, 0.05)
    assert 0.0 <= score <= 10.0


def test_investment_score_zero_market_avg():
    from app.stats import property_investment_score
    score = property_investment_score(0.05, 200.0, 0.0, 0.04)
    assert score >= 0.0


import pytest as _pytest


@_pytest.mark.parametrize("cap_rate,yield_rate", [(0.04, 0.04), (0.08, 0.07)])
def test_investment_score_parametrized(cap_rate, yield_rate):
    from app.stats import property_investment_score
    score = property_investment_score(cap_rate, 250.0, 300.0, yield_rate)
    assert 0.0 <= score <= 10.0
