"""Trading signal utilities and confidence thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalStrength(str, Enum):
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
