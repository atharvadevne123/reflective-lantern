"""Business-rule input validators for Signal-Forge."""

from __future__ import annotations

import logging

from .exceptions import ValidationError

logger = logging.getLogger(__name__)

MAX_CLOSE_PRICE: float = 1_000_000.0
MAX_VOLUME: float = 1e13
MAX_MARKET_RETURN: float = 1.0
MIN_MARKET_RETURN: float = -1.0


def validate_close(close: float) -> None:
    """Raise ValidationError if close price is unrealistically large."""
    if close > MAX_CLOSE_PRICE:
        raise ValidationError(f"close price {close} exceeds maximum {MAX_CLOSE_PRICE}")


def validate_volume(volume: float) -> None:
    """Raise ValidationError if volume is unrealistically large."""
    if volume > MAX_VOLUME:
        raise ValidationError(f"volume {volume} exceeds maximum {MAX_VOLUME}")


def validate_market_return(market_return: float) -> None:
    """Raise ValidationError if market_return is outside [-1, 1]."""
    if not (MIN_MARKET_RETURN <= market_return <= MAX_MARKET_RETURN):
        raise ValidationError(
            f"market_return {market_return} outside valid range "
            f"[{MIN_MARKET_RETURN}, {MAX_MARKET_RETURN}]"
        )


def validate_predict_input(close: float, volume: float, market_return: float) -> None:
    """Run all input validations for a /predict request."""
    validate_close(close)
    validate_volume(volume)
    validate_market_return(market_return)
    logger.debug(
        "Input validated: close=%.2f volume=%.0f market_return=%.4f",
        close, volume, market_return,
    )
