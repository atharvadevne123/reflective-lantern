"""Generate a grid health summary report from recent monitoring data."""
from __future__ import annotations

from typing import Any


def generate_grid_report(
    prediction_stats: dict[str, Any],
    drift_results: dict[str, dict],
    anomaly_count: int,
    window_hours: int = 24,
) -> dict:
    """Compose a structured grid health report from monitoring outputs.

    Args:
        prediction_stats: Dict from compute_prediction_stats.
        drift_results: Dict from check_all_features.
        anomaly_count: Number of anomalies detected in the window.
        window_hours: Monitoring window length in hours.

    Returns:
        Structured health report with status, alerts, and recommendations.
    """
    drifted = [f for f, r in drift_results.items() if r.get("drift_detected")]
    anomaly_rate = anomaly_count / max(window_hours, 1)

    status = "healthy"
    alerts: list[str] = []
    recommendations: list[str] = []

    if drifted:
        status = "warning"
        alerts.append(f"Drift detected in features: {', '.join(drifted)}")
        recommendations.append("Trigger model retraining to adapt to new distribution")

    if anomaly_rate > 0.1:
        status = "critical" if anomaly_rate > 0.3 else "warning"
        alerts.append(f"Elevated anomaly rate: {anomaly_rate:.2f}/h over {window_hours}h window")
        recommendations.append("Inspect meters with high anomaly scores for faults or tampering")

    if prediction_stats.get("std_kwh", 0) > prediction_stats.get("mean_kwh", 1) * 2:
        alerts.append("High prediction variance detected")
        recommendations.append("Review feature quality and check for data pipeline issues")

    return {
        "status": status,
        "window_hours": window_hours,
        "anomaly_count": anomaly_count,
        "anomaly_rate_per_hour": round(anomaly_rate, 4),
        "drifted_features": drifted,
        "alerts": alerts,
        "recommendations": recommendations,
        "prediction_stats": prediction_stats,
    }
