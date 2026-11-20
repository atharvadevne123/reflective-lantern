"""Tests for the alerting module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.alerting import check_drift_alert, check_risk_alert


def test_no_alert_for_low_risk():
    alert = check_risk_alert("AAPL", 0.3, "bull")
    assert alert is None


def test_warning_alert_for_high_risk():
    alert = check_risk_alert("TSLA", 0.8, "volatile")
    assert alert is not None
    assert alert.level == "WARNING"


def test_critical_alert_for_very_high_risk():
    alert = check_risk_alert("X", 0.95, "bear")
    assert alert is not None
    assert alert.level == "CRITICAL"


def test_risk_alert_contains_ticker():
    alert = check_risk_alert("NVDA", 0.82, "volatile")
    assert "NVDA" in alert.message


def test_no_drift_alert_for_high_p_value():
    alert = check_drift_alert("volatility", 0.3)
    assert alert is None


def test_drift_warning_alert():
    alert = check_drift_alert("momentum", 0.03)
    assert alert is not None
    assert alert.level == "WARNING"


def test_drift_critical_alert():
    alert = check_drift_alert("beta", 0.005)
    assert alert is not None
    assert alert.level == "CRITICAL"


@pytest.mark.parametrize("risk_score,expected_alert", [
    (0.2, None),
    (0.76, "WARNING"),
    (0.91, "CRITICAL"),
])
def test_risk_alert_thresholds(risk_score, expected_alert):
    alert = check_risk_alert("SPY", risk_score, "sideways")
    if expected_alert is None:
        assert alert is None
    else:
        assert alert.level == expected_alert
