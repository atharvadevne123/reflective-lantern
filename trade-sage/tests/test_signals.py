"""Tests for trading signal classification."""
from __future__ import annotations

import pytest

from app.signals import SignalStrength, classify_signal


def test_strong_buy_signal():
    sig = classify_signal(prediction=1, confidence=0.82, horizon_days=5)
    assert sig.strength == SignalStrength.STRONG
    assert sig.direction == "BUY"


def test_moderate_buy_signal():
    sig = classify_signal(prediction=1, confidence=0.65, horizon_days=5)
    assert sig.strength == SignalStrength.MODERATE
    assert sig.direction == "BUY"


def test_weak_buy_signal():
    sig = classify_signal(prediction=1, confidence=0.55, horizon_days=5)
    assert sig.strength == SignalStrength.WEAK
    assert sig.direction == "BUY"


def test_sell_signal_prediction_zero():
    sig = classify_signal(prediction=0, confidence=0.65, horizon_days=5)
    assert sig.direction == "SELL"


def test_signal_confidence_stored():
    sig = classify_signal(prediction=1, confidence=0.77, horizon_days=10)
    assert abs(sig.confidence - 0.77) < 1e-4


def test_classify_signal_boundary_strong():
    sig = classify_signal(prediction=1, confidence=0.75, horizon_days=5)
    assert sig.strength == SignalStrength.STRONG


def test_classify_signal_boundary_moderate():
    sig = classify_signal(prediction=1, confidence=0.60, horizon_days=5)
    assert sig.strength == SignalStrength.MODERATE


def test_trading_signal_is_actionable_strong():
    sig = classify_signal(prediction=1, confidence=0.8, horizon_days=5)
    assert sig.is_actionable is True


def test_trading_signal_is_not_actionable_weak():
    sig = classify_signal(prediction=1, confidence=0.5, horizon_days=5)
    assert sig.is_actionable is False


def test_signal_strength_values():
    assert SignalStrength.STRONG == "STRONG"
    assert SignalStrength.MODERATE == "MODERATE"
    assert SignalStrength.WEAK == "WEAK"


@pytest.mark.parametrize("pred,expected_dir", [
    (1, "BUY"),
    (0, "SELL"),
])
def test_direction_by_prediction(pred, expected_dir):
    sig = classify_signal(prediction=pred, confidence=0.8, horizon_days=5)
    assert sig.direction == expected_dir


@pytest.mark.parametrize("conf,expected_strength", [
    (0.90, SignalStrength.STRONG),
    (0.75, SignalStrength.STRONG),
    (0.74, SignalStrength.MODERATE),
    (0.60, SignalStrength.MODERATE),
    (0.59, SignalStrength.WEAK),
    (0.50, SignalStrength.WEAK),
])
def test_strength_by_confidence(conf, expected_strength):
    sig = classify_signal(prediction=1, confidence=conf, horizon_days=5)
    assert sig.strength == expected_strength
