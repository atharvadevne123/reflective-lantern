"""Reporting utilities for generating credit risk analysis reports."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def generate_risk_summary(predictions: list[float]) -> dict[str, Any]:
    """Generate a statistical summary of a batch of risk predictions.

    Args:
        predictions: List of default probability scores (0-1).

    Returns:
        Dict with count, mean, median, percentiles, and risk distribution.
    """
    if not predictions:
        return {"count": 0, "error": "no_predictions"}

    n = len(predictions)
    sorted_preds = sorted(predictions)
    mean = sum(predictions) / n
    median = sorted_preds[n // 2] if n % 2 == 1 else (sorted_preds[n // 2 - 1] + sorted_preds[n // 2]) / 2

    p25 = sorted_preds[int(n * 0.25)]
    p75 = sorted_preds[int(n * 0.75)]
    p90 = sorted_preds[int(n * 0.9)]
    p95 = sorted_preds[min(int(n * 0.95), n - 1)]

    low = sum(1 for p in predictions if p < 0.2)
    medium = sum(1 for p in predictions if 0.2 <= p < 0.5)
    high = sum(1 for p in predictions if 0.5 <= p < 0.75)
    very_high = sum(1 for p in predictions if p >= 0.75)

    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "p25": round(p25, 4),
        "p75": round(p75, 4),
        "p90": round(p90, 4),
        "p95": round(p95, 4),
        "risk_distribution": {
            "low": low,
            "medium": medium,
            "high": high,
            "very_high": very_high,
        },
        "default_rate": round(sum(1 for p in predictions if p >= 0.5) / n, 4),
    }


def generate_model_report(metrics: dict[str, Any]) -> str:
    """Format model training metrics as a readable report string.

    Args:
        metrics: Training metrics dict from model.train().

    Returns:
        Human-readable report string.
    """
    lines = [
        "=== Risk-Sense Model Report ===",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Training samples:  {metrics.get('n_samples', 'N/A')}",
        f"Positive rate:     {metrics.get('positive_rate', 0):.1%}",
        f"Train AUC-ROC:     {metrics.get('train_auc', 0):.4f}",
        f"5-fold CV AUC:     {metrics.get('cv_auc_mean', 0):.4f} ± {metrics.get('cv_auc_std', 0):.4f}",
    ]
    return "\n".join(lines)


def export_predictions_csv(records: list[dict[str, Any]]) -> str:
    """Export prediction records as CSV string.

    Args:
        records: List of prediction record dicts.

    Returns:
        CSV string with header and one row per record.
    """
    if not records:
        return "applicant_id,risk_score,risk_category,default_probability,created_at\n"

    headers = ["applicant_id", "risk_score", "risk_category", "default_probability", "created_at"]
    lines = [",".join(headers)]
    for rec in records:
        row = [str(rec.get(h, "")) for h in headers]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"
