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


def summarise_alerts(report: dict) -> str:
    """Format the alerts list from a grid report as a human-readable string.

    Args:
        report: Report dict as returned by :func:`generate_grid_report`.

    Returns:
        Formatted string with one alert per line, or 'No alerts.' if empty.
    """
    alerts = report.get("alerts", [])
    if not alerts:
        return "No alerts."
    return "\n".join(f"- {a}" for a in alerts)


def report_status_code(report: dict) -> int:
    """Map a grid report status to an HTTP-style status code.

    Returns 200 for 'healthy', 207 for 'warning', 503 for 'critical'.

    Args:
        report: Report dict as returned by :func:`generate_grid_report`.

    Returns:
        Integer status code.
    """
    status_map = {"healthy": 200, "warning": 207, "critical": 503}
    return status_map.get(str(report.get("status", "healthy")), 200)


def merge_reports(reports: list[dict]) -> dict:
    """Merge multiple grid health reports into an aggregate summary.

    Combines alert lists, sums anomaly counts, and derives the worst status.

    Args:
        reports: List of report dicts to merge.

    Returns:
        Merged report dict with combined alerts and worst-case status.
    """
    if not reports:
        return {"status": "healthy", "alerts": [], "anomaly_count": 0}
    status_priority = {"healthy": 0, "warning": 1, "critical": 2}
    worst = max(reports, key=lambda r: status_priority.get(str(r.get("status", "healthy")), 0))
    all_alerts: list[str] = []
    total_anomalies = 0
    for r in reports:
        all_alerts.extend(r.get("alerts", []))
        total_anomalies += int(r.get("anomaly_count", 0))
    return {
        "status": worst.get("status", "healthy"),
        "alerts": all_alerts,
        "anomaly_count": total_anomalies,
        "report_count": len(reports),
    }


__all__ = [
    "generate_grid_report",
    "merge_reports",
    "report_status_code",
    "summarise_alerts",
]
