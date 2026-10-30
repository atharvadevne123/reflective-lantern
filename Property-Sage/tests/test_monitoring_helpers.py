"""Tests for yield_category and anomaly_score monitoring helpers."""

import pytest

from app.monitoring import anomaly_score, yield_category


def test_yield_category_low():
    assert yield_category(0.02) == "low"


def test_yield_category_moderate():
    assert yield_category(0.04) == "moderate"


def test_yield_category_good():
    assert yield_category(0.06) == "good"


def test_yield_category_excellent():
    assert yield_category(0.09) == "excellent"


@pytest.mark.parametrize("y,expected", [
    (0.01, "low"), (0.035, "moderate"), (0.07, "good"), (0.12, "excellent")
])
def test_yield_category_parametrized(y, expected):
    assert yield_category(y) == expected


def test_anomaly_score_normal():
    ref = list(range(100))
    score = anomaly_score(50.0, ref)
    assert score < 0.2


def test_anomaly_score_extreme():
    ref = [100.0] * 50
    score = anomaly_score(1_000_000.0, ref)
    assert score == 1.0


def test_anomaly_score_empty():
    assert anomaly_score(100.0, []) == 0.0
