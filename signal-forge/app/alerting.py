"""Alerting module: emit alerts when risk score or drift thresholds are breached."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.75
CRITICAL_RISK_THRESHOLD = 0.90


@dataclass
class Alert:
    """A single alert record."""

    level: str
    message: str
    context: dict[str, Any]
    triggered_at: str


def check_risk_alert(ticker: str, risk_score: float, regime: str) -> Alert | None:
    """Return an Alert if risk_score exceeds a threshold, else None.

    Args:
        ticker: Asset ticker symbol.
        risk_score: Current portfolio risk score (0-1).
        regime: Detected market regime.

    Returns:
        Alert instance if threshold breached, else None.
    """
    if risk_score >= CRITICAL_RISK_THRESHOLD:
        level = "CRITICAL"
        message = f"{ticker} risk score {risk_score:.2f} is CRITICAL in {regime} regime"
    elif risk_score >= HIGH_RISK_THRESHOLD:
        level = "WARNING"
        message = f"{ticker} risk score {risk_score:.2f} is HIGH in {regime} regime"
    else:
        return None

    logger.warning("Alert: %s", message)
    return Alert(
        level=level,
        message=message,
        context={"ticker": ticker, "risk_score": risk_score, "regime": regime},
        triggered_at=datetime.utcnow().isoformat(),
    )


def check_drift_alert(feature: str, p_value: float) -> Alert | None:
    """Return an Alert if a KS-test p-value indicates significant drift.

    Args:
        feature: Feature name being tested.
        p_value: KS-test p-value.

    Returns:
        Alert instance if drift detected, else None.
    """
    if p_value < 0.01:
        level = "CRITICAL"
        message = f"Severe drift detected in feature '{feature}': p={p_value:.4f}"
    elif p_value < 0.05:
        level = "WARNING"
        message = f"Drift detected in feature '{feature}': p={p_value:.4f}"
    else:
        return None

    logger.warning("Drift Alert: %s", message)
    return Alert(
        level=level,
        message=message,
        context={"feature": feature, "p_value": p_value},
        triggered_at=datetime.utcnow().isoformat(),
    )
