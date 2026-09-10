"""Tests for prediction report generation."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest

from app.reporting import format_prediction_report, report_to_json, log_report_summary


FEATURES = {"volatility": 0.18, "momentum": 0.04, "volume_ratio": 1.1, "correlation": 0.6, "beta": 1.05}


def test_format_prediction_report_keys():
    report = format_prediction_report("AAPL", "bull", 0.8, 0.3, FEATURES, [])
    required = {"ticker", "regime", "confidence", "risk_score", "risk_tier", "features", "generated_at"}
    assert required.issubset(report.keys())


def test_format_prediction_report_risk_tier_low():
    report = format_prediction_report("AAPL", "bull", 0.8, 0.1, FEATURES, [])
    assert report["risk_tier"] == "low"


def test_format_prediction_report_risk_tier_critical():
    report = format_prediction_report("X", "volatile", 0.9, 0.9, FEATURES, [])
    assert report["risk_tier"] == "critical"


def test_report_to_json_valid():
    report = format_prediction_report("SPY", "sideways", 0.6, 0.4, FEATURES, [])
    j = report_to_json(report)
    parsed = json.loads(j)
    assert parsed["ticker"] == "SPY"


def test_report_to_json_empty():
    assert report_to_json({}) == "{}"


@pytest.mark.parametrize("regime,score", [
    ("bull", 0.2),
    ("bear", 0.8),
    ("sideways", 0.35),
    ("volatile", 0.85),
])
def test_format_report_all_regimes(regime, score):
    report = format_prediction_report("TEST", regime, 0.7, score, FEATURES, [])
    assert report["regime"] == regime
    assert report["risk_score"] == pytest.approx(score, abs=0.001)
