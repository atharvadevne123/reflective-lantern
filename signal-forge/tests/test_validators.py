"""Tests for business-rule input validators."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.exceptions import ValidationError
from app.validators import (
    validate_close,
    validate_market_return,
    validate_predict_input,
    validate_volume,
)


def test_validate_close_normal():
    validate_close(195.0)  # should not raise


def test_validate_close_too_large():
    with pytest.raises(ValidationError):
        validate_close(2_000_000.0)


def test_validate_volume_normal():
    validate_volume(80_000_000.0)  # should not raise


def test_validate_volume_too_large():
    with pytest.raises(ValidationError):
        validate_volume(1e14)


@pytest.mark.parametrize("ret", [-1.0, 0.0, 0.5, 1.0])
def test_validate_market_return_valid(ret):
    validate_market_return(ret)  # should not raise


@pytest.mark.parametrize("ret", [-1.5, 1.1, 2.0])
def test_validate_market_return_invalid(ret):
    with pytest.raises(ValidationError):
        validate_market_return(ret)


def test_validate_predict_input_valid():
    validate_predict_input(100.0, 1_000_000.0, 0.005)  # should not raise


def test_validate_predict_input_bad_close():
    with pytest.raises(ValidationError):
        validate_predict_input(5_000_000.0, 1_000.0, 0.0)
