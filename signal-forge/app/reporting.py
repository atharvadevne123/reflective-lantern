"""Prediction report generation utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def format_prediction_report(
    ticker: str,
    regime: str,
    confidence: float,
    risk_score: float,
    features: dict[str, float],
    similar_periods: list[dict[str, Any]],
    model_version: str = "1.0.0",
) -> dict[str, Any]:
    """Compose a structured prediction report dict.

    Args:
        ticker: Asset ticker symbol.
        regime: Detected market regime label.
        confidence: Ensemble confidence score (0-1).
        risk_score: Portfolio risk score (0-1).
        features: Dict of extracted feature values.
        similar_periods: FAISS nearest-neighbour results.
        model_version: Version string of the deployed model.

    Returns:
        Report dict suitable for API response or logging.
    """
    from app.cache import get_risk_tier
    risk_tier = get_risk_tier(round(risk_score, 2))
    return {
        "ticker": ticker,
        "regime": regime,
        "confidence": round(confidence, 4),
        "risk_score": round(risk_score, 4),
        "risk_tier": risk_tier,
        "features": {k: round(v, 4) for k, v in features.items()},
        "similar_periods_count": len(similar_periods),
        "similar_periods": similar_periods[:3],
        "model_version": model_version,
        "generated_at": datetime.utcnow().isoformat(),
    }


def report_to_json(report: dict[str, Any], indent: int = 2) -> str:
    """Serialise a prediction report to JSON string.

    Args:
        report: Prediction report dict.
        indent: JSON indentation level.

    Returns:
        JSON-formatted string.
    """
    try:
        return json.dumps(report, indent=indent, default=str)
    except Exception as e:
        logger.error("Failed to serialise report: %s", e)
        return "{}"


def log_report_summary(report: dict[str, Any]) -> None:
    """Emit a one-line log summary of a prediction report."""
    logger.info(
        "ticker=%s regime=%s confidence=%.3f risk_score=%.3f risk_tier=%s",
        report.get("ticker"),
        report.get("regime"),
        report.get("confidence", 0.0),
        report.get("risk_score", 0.0),
        report.get("risk_tier"),
    )
