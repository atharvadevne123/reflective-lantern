"""Extended input validator tests for signal-forge — second pass."""
from __future__ import annotations

import pytest

from app.exceptions import ValidationError
from app.validators import (
    MAX_CLOSE_PRICE,
    MAX_MARKET_RETURN,
    MAX_VOLUME,
    MIN_MARKET_RETURN,
    validate_close,
    validate_market_return,
    validate_predict_input,
    validate_volume,
)


def test_validate_close_valid_does_not_raise() -> None:
    validate_close(100.0)


def test_validate_close_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        validate_close(MAX_CLOSE_PRICE + 1.0)


def test_validate_close_at_max_does_not_raise() -> None:
    validate_close(MAX_CLOSE_PRICE)


def test_validate_volume_valid_does_not_raise() -> None:
    validate_volume(1_000_000.0)


def test_validate_volume_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        validate_volume(MAX_VOLUME + 1.0)


def test_validate_volume_at_max_does_not_raise() -> None:
    validate_volume(MAX_VOLUME)


def test_validate_market_return_zero_does_not_raise() -> None:
    validate_market_return(0.0)


def test_validate_market_return_at_min_does_not_raise() -> None:
    validate_market_return(MIN_MARKET_RETURN)


def test_validate_market_return_at_max_does_not_raise() -> None:
    validate_market_return(MAX_MARKET_RETURN)


def test_validate_market_return_below_min_raises() -> None:
    with pytest.raises(ValidationError):
        validate_market_return(MIN_MARKET_RETURN - 0.001)


def test_validate_market_return_above_max_raises() -> None:
    with pytest.raises(ValidationError):
        validate_market_return(MAX_MARKET_RETURN + 0.001)


@pytest.mark.parametrize("close", [0.01, 1.0, 100.0, 50000.0])
def test_validate_close_various_valid_prices(close: float) -> None:
    validate_close(close)


@pytest.mark.parametrize("volume", [0.0, 1.0, 1e6, MAX_VOLUME])
def test_validate_volume_various_valid_volumes(volume: float) -> None:
    validate_volume(volume)


@pytest.mark.parametrize("market_return", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_validate_market_return_various_valid(market_return: float) -> None:
    validate_market_return(market_return)


def test_validate_predict_input_valid_does_not_raise() -> None:
    validate_predict_input(close=150.0, volume=1_000_000.0, market_return=0.01)


def test_validate_predict_input_bad_close_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=MAX_CLOSE_PRICE + 1, volume=1_000.0, market_return=0.0)


def test_validate_predict_input_bad_volume_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=100.0, volume=MAX_VOLUME + 1, market_return=0.0)


def test_validate_predict_input_bad_market_return_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=100.0, volume=1_000.0, market_return=2.0)
