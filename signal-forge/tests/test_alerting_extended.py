"""Extended tests for signal-forge app/alerting.py."""

from __future__ import annotations

import pytest

from app.alerting import (
    CRITICAL_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    Alert,
    check_drift_alert,
    check_risk_alert,
)


class TestAlert:
    def test_create_alert(self) -> None:
        a = Alert(level="WARNING", message="test", context={}, triggered_at="2024-01-01")
        assert a.level == "WARNING"
        assert a.message == "test"

    def test_alert_has_triggered_at(self) -> None:
        a = Alert(level="INFO", message="msg", context={"k": 1}, triggered_at="2024-01-01T00:00:00")
        assert a.triggered_at is not None


class TestCheckRiskAlert:
    def test_below_threshold_returns_none(self) -> None:
        result = check_risk_alert("AAPL", 0.3, "bull")
        assert result is None

    def test_at_high_threshold_returns_warning(self) -> None:
        result = check_risk_alert("AAPL", HIGH_RISK_THRESHOLD, "bear")
        assert result is not None
        assert result.level == "WARNING"

    def test_at_critical_threshold_returns_critical(self) -> None:
        result = check_risk_alert("AAPL", CRITICAL_RISK_THRESHOLD, "volatile")
        assert result is not None
        assert result.level == "CRITICAL"

    def test_above_critical_returns_critical(self) -> None:
        result = check_risk_alert("TSLA", 0.99, "volatile")
        assert result is not None
        assert result.level == "CRITICAL"

    def test_ticker_in_message(self) -> None:
        result = check_risk_alert("MSFT", HIGH_RISK_THRESHOLD, "bear")
        assert result is not None
        assert "MSFT" in result.message

    def test_context_has_ticker(self) -> None:
        result = check_risk_alert("GOOG", 0.8, "bull")
        assert result is not None
        assert result.context["ticker"] == "GOOG"

    def test_context_has_risk_score(self) -> None:
        result = check_risk_alert("AMZN", 0.8, "sideways")
        assert result is not None
        assert result.context["risk_score"] == pytest.approx(0.8)


class TestCheckDriftAlert:
    def test_high_p_value_no_alert(self) -> None:
        result = check_drift_alert("volatility", 0.5)
        assert result is None

    def test_low_p_value_returns_alert(self) -> None:
        result = check_drift_alert("momentum", 0.01)
        assert result is not None

    def test_feature_name_in_message(self) -> None:
        result = check_drift_alert("volume_ratio", 0.001)
        assert result is not None
        assert "volume_ratio" in result.message


class TestThresholdConstants:
    def test_critical_gt_high(self) -> None:
        assert CRITICAL_RISK_THRESHOLD > HIGH_RISK_THRESHOLD

    def test_thresholds_in_range(self) -> None:
        assert 0.0 < HIGH_RISK_THRESHOLD < 1.0
        assert 0.0 < CRITICAL_RISK_THRESHOLD <= 1.0
