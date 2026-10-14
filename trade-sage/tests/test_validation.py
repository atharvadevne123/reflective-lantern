"""Tests for Pydantic validation models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.validation import OHLCVRow, PortfolioRiskRequest, PredictRequest


def _row(o=100.0, h=105.0, lo=95.0, c=102.0, v=1_000_000.0):
    return {"open": o, "high": h, "low": lo, "close": c, "volume": v}


def _rows(n=70):
    return [_row() for _ in range(n)]


def test_ohlcv_row_valid():
    row = OHLCVRow(**_row())
    assert row.close == 102.0


def test_ohlcv_row_high_lt_low_invalid():
    with pytest.raises(ValidationError):
        OHLCVRow(open=100, high=90, low=95, close=100, volume=1e6)


def test_ohlcv_row_zero_price_invalid():
    with pytest.raises(ValidationError):
        OHLCVRow(open=0, high=1, low=0.5, close=1, volume=1e6)


def test_predict_request_ticker_uppercased():
    req = PredictRequest(ticker="aapl", horizon_days=5, ohlcv=_rows())
    assert req.ticker == "AAPL"


def test_predict_request_too_few_rows():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="AAPL", horizon_days=5, ohlcv=_rows(30))


def test_predict_request_horizon_out_of_range():
    with pytest.raises(ValidationError):
        PredictRequest(ticker="AAPL", horizon_days=0, ohlcv=_rows())


def test_portfolio_weights_not_summing():
    with pytest.raises(ValidationError):
        PortfolioRiskRequest(holdings=[
            {"ticker": "AAPL", "weight": 0.3, "ohlcv": _rows()},
            {"ticker": "GOOG", "weight": 0.3, "ohlcv": _rows()},
        ])


def test_portfolio_valid():
    req = PortfolioRiskRequest(holdings=[
        {"ticker": "AAPL", "weight": 0.6, "ohlcv": _rows()},
        {"ticker": "GOOG", "weight": 0.4, "ohlcv": _rows()},
    ])
    assert len(req.holdings) == 2
