"""Tests for lru_cache helpers."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.cache import get_regime_label, get_risk_tier


@pytest.mark.parametrize("idx,expected", [
    (0, "bull"),
    (1, "bear"),
    (2, "sideways"),
    (3, "volatile"),
])
def test_get_regime_label(idx, expected):
    assert get_regime_label(idx) == expected


def test_get_regime_label_unknown():
    assert get_regime_label(99) == "unknown"


@pytest.mark.parametrize("score,expected", [
    (0.1, "low"),
    (0.3, "moderate"),
    (0.6, "high"),
    (0.9, "critical"),
])
def test_get_risk_tier(score, expected):
    assert get_risk_tier(score) == expected


def test_get_regime_label_cached():
    get_regime_label.cache_clear()
    get_regime_label(0)
    get_regime_label(0)
    info = get_regime_label.cache_info()
    assert info.hits >= 1
