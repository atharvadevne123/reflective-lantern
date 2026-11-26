"""Extended validator tests for signal-forge with parametrize and boundary coverage."""

from __future__ import annotations

import pytest

from app.exceptions import ValidationError
from app.validators import (
    MAX_CLOSE_PRICE,
    MAX_VOLUME,
    validate_close,
    validate_market_return,
    validate_predict_input,
    validate_volume,
)


@pytest.mark.parametrize("valid_close", [0.0, 1.0, 100.0, 999_999.99, MAX_CLOSE_PRICE])
def test_validate_close_valid_values(valid_close: float) -> None:
    validate_close(valid_close)


@pytest.mark.parametrize("bad_close", [MAX_CLOSE_PRICE + 1, 2_000_000.0, 1e9])
def test_validate_close_too_large_raises(bad_close: float) -> None:
    with pytest.raises(ValidationError, match="close price"):
        validate_close(bad_close)


def test_validate_close_exactly_at_max_does_not_raise() -> None:
    validate_close(MAX_CLOSE_PRICE)


@pytest.mark.parametrize("valid_vol", [0.0, 1.0, 1e6, 1e12, MAX_VOLUME])
def test_validate_volume_valid_values(valid_vol: float) -> None:
    validate_volume(valid_vol)


@pytest.mark.parametrize("bad_vol", [MAX_VOLUME + 1, 1e14, 1e15])
def test_validate_volume_too_large_raises(bad_vol: float) -> None:
    with pytest.raises(ValidationError, match="volume"):
        validate_volume(bad_vol)


@pytest.mark.parametrize("valid_ret", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_validate_market_return_valid(valid_ret: float) -> None:
    validate_market_return(valid_ret)


@pytest.mark.parametrize("bad_ret", [-1.001, -2.0, 1.001, 2.0])
def test_validate_market_return_out_of_range_raises(bad_ret: float) -> None:
    with pytest.raises(ValidationError, match="market_return"):
        validate_market_return(bad_ret)


def test_validate_market_return_exactly_minus_one() -> None:
    validate_market_return(-1.0)


def test_validate_market_return_exactly_one() -> None:
    validate_market_return(1.0)


def test_validate_predict_input_all_valid() -> None:
    validate_predict_input(close=100.0, volume=1_000_000.0, market_return=0.02)


def test_validate_predict_input_bad_close_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=MAX_CLOSE_PRICE + 1, volume=1_000.0, market_return=0.0)


def test_validate_predict_input_bad_volume_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=100.0, volume=MAX_VOLUME + 1, market_return=0.0)


def test_validate_predict_input_bad_market_return_raises() -> None:
    with pytest.raises(ValidationError):
        validate_predict_input(close=100.0, volume=1_000.0, market_return=1.5)


@pytest.mark.parametrize(
    "close,volume,ret",
    [
        (0.01, 1.0, 0.0),
        (500.0, 1e10, -0.99),
        (MAX_CLOSE_PRICE, MAX_VOLUME, 1.0),
    ],
)
def test_validate_predict_input_boundary_valid(close: float, volume: float, ret: float) -> None:
    validate_predict_input(close=close, volume=volume, market_return=ret)


def test_validation_error_is_subclass_of_exception() -> None:
    with pytest.raises(Exception):
        validate_close(MAX_CLOSE_PRICE + 100)
