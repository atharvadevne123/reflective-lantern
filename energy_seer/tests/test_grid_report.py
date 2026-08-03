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


class TestSummariseAlerts:
    def test_no_alerts(self) -> None:
        from energy_seer.app.grid_report import summarise_alerts
        assert summarise_alerts({"alerts": []}) == "No alerts."

    def test_formats_alerts(self) -> None:
        from energy_seer.app.grid_report import summarise_alerts
        result = summarise_alerts({"alerts": ["Alert A", "Alert B"]})
        assert "- Alert A" in result
        assert "- Alert B" in result

    def test_missing_key(self) -> None:
        from energy_seer.app.grid_report import summarise_alerts
        assert summarise_alerts({}) == "No alerts."


class TestReportStatusCode:
    def test_healthy(self) -> None:
        from energy_seer.app.grid_report import report_status_code
        assert report_status_code({"status": "healthy"}) == 200

    def test_warning(self) -> None:
        from energy_seer.app.grid_report import report_status_code
        assert report_status_code({"status": "warning"}) == 207

    def test_critical(self) -> None:
        from energy_seer.app.grid_report import report_status_code
        assert report_status_code({"status": "critical"}) == 503

    def test_unknown_defaults_200(self) -> None:
        from energy_seer.app.grid_report import report_status_code
        assert report_status_code({"status": "unknown"}) == 200


class TestMergeReports:
    def test_empty_list(self) -> None:
        from energy_seer.app.grid_report import merge_reports
        result = merge_reports([])
        assert result["status"] == "healthy"
        assert result["anomaly_count"] == 0

    def test_worst_status_wins(self) -> None:
        from energy_seer.app.grid_report import merge_reports
        reports = [
            {"status": "healthy", "alerts": [], "anomaly_count": 0},
            {"status": "critical", "alerts": ["x"], "anomaly_count": 5},
        ]
        result = merge_reports(reports)
        assert result["status"] == "critical"

    def test_alert_combination(self) -> None:
        from energy_seer.app.grid_report import merge_reports
        reports = [
            {"status": "warning", "alerts": ["A"], "anomaly_count": 1},
            {"status": "warning", "alerts": ["B"], "anomaly_count": 2},
        ]
        result = merge_reports(reports)
        assert "A" in result["alerts"] and "B" in result["alerts"]
        assert result["anomaly_count"] == 3
