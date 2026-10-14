"""Tests for trading signal classification utilities."""
from __future__ import annotations

import pytest

from app.signals import SignalStrength, TradingSignal, classify_signal


@pytest.mark.parametrize("pred,direction", [(1, "BUY"), (0, "SELL")])
def test_classify_direction(pred, direction):
    sig = classify_signal(pred, 0.70, 5)
    assert sig.direction == direction


@pytest.mark.parametrize("conf,expected", [
    (0.80, SignalStrength.STRONG),
    (0.65, SignalStrength.MODERATE),
    (0.52, SignalStrength.WEAK),
])
def test_classify_strength(conf, expected):
    sig = classify_signal(1, conf, 5)
    assert sig.strength == expected


def test_is_actionable_strong():
    sig = classify_signal(1, 0.85, 5)
    assert sig.is_actionable


def test_is_actionable_weak():
    sig = classify_signal(0, 0.55, 5)
    assert not sig.is_actionable


def test_confidence_rounded():
    sig = classify_signal(1, 0.74321, 5)
    assert sig.confidence == 0.7432
