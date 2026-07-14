"""Tests for grid health report generator."""
from __future__ import annotations


class TestGenerateGridReport:
    def test_healthy_status_no_issues(self):
        from app.grid_report import generate_grid_report
        result = generate_grid_report(
            prediction_stats={"mean_kwh": 4.0, "std_kwh": 0.5},
            drift_results={},
            anomaly_count=0,
        )
        assert result["status"] == "healthy"
        assert result["alerts"] == []

    def test_warning_on_drift(self):
        from app.grid_report import generate_grid_report
        result = generate_grid_report(
            prediction_stats={"mean_kwh": 4.0, "std_kwh": 0.5},
            drift_results={"consumption_kwh": {"drift_detected": True}},
            anomaly_count=0,
        )
        assert result["status"] == "warning"
        assert "consumption_kwh" in result["drifted_features"]

    def test_critical_on_high_anomaly_rate(self):
        from app.grid_report import generate_grid_report
        result = generate_grid_report(
            prediction_stats={"mean_kwh": 4.0, "std_kwh": 0.5},
            drift_results={},
            anomaly_count=10,
            window_hours=24,
        )
        assert result["status"] in {"warning", "critical"}

    def test_report_has_required_keys(self):
        from app.grid_report import generate_grid_report
        result = generate_grid_report({}, {}, 0)
        for key in ["status", "alerts", "recommendations", "anomaly_count"]:
            assert key in result

    def test_recommendations_on_drift(self):
        from app.grid_report import generate_grid_report
        result = generate_grid_report(
            {},
            {"temp": {"drift_detected": True}},
            0,
        )
        assert len(result["recommendations"]) > 0
