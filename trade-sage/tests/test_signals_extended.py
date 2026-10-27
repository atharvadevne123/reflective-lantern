"""Extended tests for trade-sage app/signals.py."""

from __future__ import annotations

from app.signals import SignalStrength, TradingSignal, classify_signal


class TestSignalStrength:
    def test_strong_value(self) -> None:
        assert SignalStrength.STRONG == "STRONG"

    def test_moderate_value(self) -> None:
        assert SignalStrength.MODERATE == "MODERATE"

    def test_weak_value(self) -> None:
        assert SignalStrength.WEAK == "WEAK"

    def test_is_str_subtype(self) -> None:
        assert isinstance(SignalStrength.STRONG, str)


class TestTradingSignal:
    def test_create_buy_signal(self) -> None:
        sig = TradingSignal(
            direction="BUY",
            confidence=0.8,
            strength=SignalStrength.STRONG,
            horizon_days=5,
        )
        assert sig.direction == "BUY"

    def test_create_sell_signal(self) -> None:
        sig = TradingSignal(
            direction="SELL",
            confidence=0.65,
            strength=SignalStrength.MODERATE,
            horizon_days=3,
        )
        assert sig.direction == "SELL"

    def test_strong_is_actionable(self) -> None:
        sig = TradingSignal("BUY", 0.9, SignalStrength.STRONG, 5)
        assert sig.is_actionable is True

    def test_moderate_is_actionable(self) -> None:
        sig = TradingSignal("BUY", 0.65, SignalStrength.MODERATE, 5)
        assert sig.is_actionable is True

    def test_weak_is_not_actionable(self) -> None:
        sig = TradingSignal("BUY", 0.5, SignalStrength.WEAK, 5)
        assert sig.is_actionable is False


class TestClassifySignal:
    def test_high_confidence_buy_is_strong(self) -> None:
        sig = classify_signal(prediction=1, confidence=0.80, horizon_days=5)
        assert sig.direction == "BUY"
        assert sig.strength == SignalStrength.STRONG

    def test_medium_confidence_is_moderate(self) -> None:
        sig = classify_signal(prediction=1, confidence=0.65, horizon_days=5)
        assert sig.strength == SignalStrength.MODERATE

    def test_low_confidence_is_weak(self) -> None:
        sig = classify_signal(prediction=0, confidence=0.55, horizon_days=5)
        assert sig.strength == SignalStrength.WEAK
        assert sig.direction == "SELL"

    def test_prediction_zero_is_sell(self) -> None:
        sig = classify_signal(prediction=0, confidence=0.8, horizon_days=5)
        assert sig.direction == "SELL"

    def test_prediction_one_is_buy(self) -> None:
        sig = classify_signal(prediction=1, confidence=0.8, horizon_days=5)
        assert sig.direction == "BUY"

    def test_confidence_rounded(self) -> None:
        sig = classify_signal(prediction=1, confidence=0.756789, horizon_days=5)
        assert len(str(sig.confidence).split(".")[-1]) <= 4

    def test_horizon_days_preserved(self) -> None:
        sig = classify_signal(prediction=1, confidence=0.8, horizon_days=10)
        assert sig.horizon_days == 10
