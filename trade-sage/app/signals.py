"""Trading signal utilities and confidence thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SignalStrength(StrEnum):
    """Qualitative signal strength based on model confidence."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class TradingSignal:
    """Structured trading signal with metadata.

    Attributes:
        direction: 'BUY' or 'SELL'.
        confidence: Model probability for the predicted class.
        strength: Qualitative strength bucket.
        horizon_days: Prediction horizon in trading days.
    """

    direction: str
    confidence: float
    strength: SignalStrength
    horizon_days: int

    @property
    def is_actionable(self) -> bool:
        """Return True when confidence exceeds the MODERATE threshold."""
        return self.strength in (SignalStrength.STRONG, SignalStrength.MODERATE)


def classify_signal(prediction: int, confidence: float, horizon_days: int) -> TradingSignal:
    """Convert raw model output to a structured trading signal.

    Args:
        prediction: Binary class (1=bullish, 0=bearish).
        confidence: Probability for the predicted class.
        horizon_days: Forecast window in trading days.

    Returns:
        TradingSignal with direction, confidence, and strength.
    """
    direction = "BUY" if prediction == 1 else "SELL"
    if confidence >= 0.75:
        strength = SignalStrength.STRONG
    elif confidence >= 0.60:
        strength = SignalStrength.MODERATE
    else:
        strength = SignalStrength.WEAK
    return TradingSignal(
        direction=direction,
        confidence=round(confidence, 4),
        strength=strength,
        horizon_days=horizon_days,
    )


def aggregate_signals(signals: list[TradingSignal]) -> dict[str, int | float]:
    """Summarise a list of trading signals into vote counts and mean confidence.

    Args:
        signals: Non-empty list of TradingSignal instances.

    Returns:
        Dict with keys ``buy_votes``, ``sell_votes``, ``consensus``,
        and ``mean_confidence``.  ``consensus`` is 'BUY', 'SELL', or 'NEUTRAL'.

    Raises:
        ValueError: If signals is empty.
    """
    if not signals:
        raise ValueError("signals list must not be empty")
    buy_votes = sum(1 for s in signals if s.direction == "BUY")
    sell_votes = len(signals) - buy_votes
    mean_conf = round(sum(s.confidence for s in signals) / len(signals), 4)
    if buy_votes > sell_votes:
        consensus = "BUY"
    elif sell_votes > buy_votes:
        consensus = "SELL"
    else:
        consensus = "NEUTRAL"
    return {
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "consensus": consensus,
        "mean_confidence": mean_conf,
    }
