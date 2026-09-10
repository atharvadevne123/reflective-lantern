"""Tests for Pydantic v2 request/response schemas."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError

from app.schemas import PredictRequest, PredictResponse


def test_predict_request_normalises_ticker():
    req = PredictRequest(ticker="aapl", close=100.0, volume=1_000_000.0)
    assert req.ticker == "AAPL"


def test_predict_request_strips_whitespace():
    req = PredictRequest(ticker=" TSLA ", close=250.0, volume=5_000_000.0)
    assert req.ticker == "TSLA"


def test_predict_request_defaults_market_return():
    req = PredictRequest(ticker="SPY", close=500.0, volume=100_000_000.0)
    assert req.market_return == 0.0


def test_predict_request_rejects_zero_close():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="X", close=0.0, volume=1_000.0)


def test_predict_request_rejects_negative_close():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="X", close=-1.0, volume=1_000.0)


def test_predict_request_rejects_empty_ticker():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="", close=100.0, volume=1_000.0)


def test_predict_request_rejects_long_ticker():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="TOOLONGTICKER", close=100.0, volume=1_000.0)


@pytest.mark.parametrize("ticker,close,volume", [
    ("BTC-USD", 60000.0, 1_000_000.0),
    ("ETH", 3000.0, 500_000.0),
    ("NVDA", 900.0, 20_000_000.0),
])
def test_predict_request_valid_tickers(ticker, close, volume):
    req = PredictRequest(ticker=ticker, close=close, volume=volume)
    assert req.close == close
