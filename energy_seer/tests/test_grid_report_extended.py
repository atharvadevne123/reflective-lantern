"""Extended tests for energy_seer app/grid_report.py."""

from __future__ import annotations

import pytest

from app.grid_report import (
    alert_frequency,
    compute_anomaly_rate,
    generate_grid_report,
    merge_reports,
    report_health_score,
    report_status_code,
    report_trend,
    summarise_alerts,
)


class TestGenerateGridReportExtended:
    def test_healthy_has_no_alerts(self) -> None:
        result = generate_grid_report({}, {}, anomaly_count=0)
        assert result["status"] == "healthy"
        assert result["alerts"] == []

    def test_drift_sets_warning(self) -> None:
        result = generate_grid_report({}, {"feat": {"drift_detected": True}}, 0)
        assert result["status"] == "warning"

    def test_low_anomaly_rate_stays_warning(self) -> None:
        result = generate_grid_report({}, {}, anomaly_count=5, window_hours=100)
        assert result["status"] == "healthy"

    def test_critical_on_very_high_anomaly_rate(self) -> None:
        result = generate_grid_report({}, {}, anomaly_count=100, window_hours=10)
        assert result["status"] == "critical"

    def test_high_variance_adds_alert(self) -> None:
        result = generate_grid_report({"mean_kwh": 1.0, "std_kwh": 5.0}, {}, 0)
        assert any("variance" in a.lower() for a in result["alerts"])

    def test_window_hours_stored_in_report(self) -> None:
        result = generate_grid_report({}, {}, 0, window_hours=48)
        assert result["window_hours"] == 48

    def test_drifted_features_list_correct(self) -> None:
        drift = {"a": {"drift_detected": True}, "b": {"drift_detected": False}}
        result = generate_grid_report({}, drift, 0)
        assert "a" in result["drifted_features"]
        assert "b" not in result["drifted_features"]

    def test_anomaly_rate_computed(self) -> None:
        result = generate_grid_report({}, {}, anomaly_count=12, window_hours=24)
        assert result["anomaly_rate_per_hour"] == pytest.approx(0.5)

    @pytest.mark.parametrize("count,hours", [(0, 24), (10, 100), (5, 50)])
    def test_anomaly_rate_parametrized(self, count: int, hours: int) -> None:
        result = generate_grid_report({}, {}, anomaly_count=count, window_hours=hours)
        assert abs(result["anomaly_rate_per_hour"] - count / hours) < 0.001


class TestSummariseAlertsExtended:
    def test_no_alerts_returns_no_alerts_message(self) -> None:
        assert summarise_alerts({"alerts": []}) == "No alerts."

    def test_missing_alerts_key_returns_no_alerts(self) -> None:
        assert summarise_alerts({}) == "No alerts."

    def test_single_alert_formatted(self) -> None:
        result = summarise_alerts({"alerts": ["Drift detected"]})
        assert "Drift detected" in result
        assert result.startswith("- ")

    def test_multiple_alerts_one_per_line(self) -> None:
        alerts = ["Alert 1", "Alert 2", "Alert 3"]
        result = summarise_alerts({"alerts": alerts})
        lines = result.strip().split("\n")
        assert len(lines) == 3

    def test_all_lines_start_with_dash(self) -> None:
        alerts = ["A", "B", "C"]
        result = summarise_alerts({"alerts": alerts})
        for line in result.split("\n"):
            assert line.startswith("- ")


class TestReportStatusCodeExtended:
    def test_healthy_returns_200(self) -> None:
        assert report_status_code({"status": "healthy"}) == 200

    def test_warning_returns_207(self) -> None:
        assert report_status_code({"status": "warning"}) == 207

    def test_critical_returns_503(self) -> None:
        assert report_status_code({"status": "critical"}) == 503

    def test_missing_status_returns_200(self) -> None:
        assert report_status_code({}) == 200

    def test_unknown_status_returns_200(self) -> None:
        assert report_status_code({"status": "unknown"}) == 200

    @pytest.mark.parametrize(
        "status,expected",
        [("healthy", 200), ("warning", 207), ("critical", 503)],
    )
    def test_all_status_codes(self, status: str, expected: int) -> None:
        assert report_status_code({"status": status}) == expected


class TestMergeReportsExtended:
    def test_empty_list_returns_healthy(self) -> None:
        result = merge_reports([])
        assert result["status"] == "healthy"
        assert result["anomaly_count"] == 0

    def test_all_healthy_merges_to_healthy(self) -> None:
        reports = [{"status": "healthy", "alerts": [], "anomaly_count": 0}] * 3
        result = merge_reports(reports)
        assert result["status"] == "healthy"

    def test_critical_dominates(self) -> None:
        reports = [
            {"status": "healthy", "alerts": [], "anomaly_count": 0},
            {"status": "critical", "alerts": ["Big problem"], "anomaly_count": 10},
        ]
        result = merge_reports(reports)
        assert result["status"] == "critical"

    def test_anomaly_counts_summed(self) -> None:
        reports = [
            {"status": "healthy", "alerts": [], "anomaly_count": 3},
            {"status": "healthy", "alerts": [], "anomaly_count": 7},
        ]
        result = merge_reports(reports)
        assert result["anomaly_count"] == 10

    def test_alerts_combined(self) -> None:
        reports = [
            {"status": "warning", "alerts": ["A"], "anomaly_count": 0},
            {"status": "warning", "alerts": ["B"], "anomaly_count": 0},
        ]
        result = merge_reports(reports)
        assert "A" in result["alerts"]
        assert "B" in result["alerts"]


class TestReportTrendAndAlertFrequency:
    def test_empty_reports_returns_defaults(self) -> None:
        result = report_trend([])
        assert result["escalating"] is False
        assert result["anomaly_total"] == 0

    def test_escalating_true_when_worsening(self) -> None:
        reports = [{"status": "healthy", "anomaly_count": 0}, {"status": "critical", "anomaly_count": 5}]
        result = report_trend(reports)
        assert result["escalating"] is True

    def test_not_escalating_when_stable(self) -> None:
        reports = [{"status": "warning", "anomaly_count": 2}, {"status": "warning", "anomaly_count": 3}]
        result = report_trend(reports)
        assert result["escalating"] is False

    def test_anomaly_total_summed(self) -> None:
        reports = [{"status": "healthy", "anomaly_count": 2}, {"status": "healthy", "anomaly_count": 5}]
        result = report_trend(reports)
        assert result["anomaly_total"] == 7

    def test_alert_frequency_counts_occurrences(self) -> None:
        reports = [
            {"alerts": ["A", "B"]},
            {"alerts": ["A", "C"]},
            {"alerts": ["A"]},
        ]
        freq = alert_frequency(reports)
        assert freq["A"] == 3
        assert freq["B"] == 1

    def test_compute_anomaly_rate_zero_hours(self) -> None:
        assert compute_anomaly_rate(10, 0) == 0.0

    def test_compute_anomaly_rate_basic(self) -> None:
        assert compute_anomaly_rate(24, 24) == pytest.approx(1.0)


class TestReportHealthScore:
    def test_healthy_no_issues_scores_100(self) -> None:
        report = {
            "status": "healthy",
            "alerts": [],
            "drifted_features": [],
            "anomaly_rate_per_hour": 0.0,
        }
        assert report_health_score(report) == pytest.approx(100.0)

    def test_critical_status_lowers_score(self) -> None:
        report = {
            "status": "critical",
            "alerts": [],
            "drifted_features": [],
            "anomaly_rate_per_hour": 0.0,
        }
        assert report_health_score(report) < 50.0

    def test_alerts_reduce_score(self) -> None:
        no_alert = {"status": "healthy", "alerts": [], "drifted_features": [], "anomaly_rate_per_hour": 0.0}
        with_alert = {"status": "healthy", "alerts": ["X", "Y"], "drifted_features": [], "anomaly_rate_per_hour": 0.0}
        assert report_health_score(with_alert) < report_health_score(no_alert)

    def test_score_clamped_to_zero(self) -> None:
        report = {
            "status": "critical",
            "alerts": ["A"] * 20,
            "drifted_features": ["x"] * 10,
            "anomaly_rate_per_hour": 10.0,
        }
        assert report_health_score(report) >= 0.0

    def test_score_clamped_to_100(self) -> None:
        report = {"status": "healthy", "alerts": [], "drifted_features": [], "anomaly_rate_per_hour": 0.0}
        assert report_health_score(report) <= 100.0

    def test_drifted_features_reduce_score(self) -> None:
        clean = {"status": "healthy", "alerts": [], "drifted_features": [], "anomaly_rate_per_hour": 0.0}
        drifted = {"status": "healthy", "alerts": [], "drifted_features": ["a", "b", "c"], "anomaly_rate_per_hour": 0.0}
        assert report_health_score(drifted) < report_health_score(clean)


class TestReportTrend:
    def test_empty_returns_defaults(self) -> None:
        result = report_trend([])
        assert result["anomaly_total"] == 0
        assert result["escalating"] is False

    def test_escalating_healthy_to_critical(self) -> None:
        reports = [{"status": "healthy"}, {"status": "critical"}]
        result = report_trend(reports)
        assert result["escalating"] is True

    def test_no_escalation_same_status(self) -> None:
        reports = [{"status": "warning"}, {"status": "warning"}]
        result = report_trend(reports)
        assert result["escalating"] is False

    def test_anomaly_total_summed(self) -> None:
        reports = [{"status": "healthy", "anomaly_count": 3}, {"status": "healthy", "anomaly_count": 5}]
        result = report_trend(reports)
        assert result["anomaly_total"] == 8


class TestAlertFrequency:
    def test_empty_returns_empty(self) -> None:
        assert alert_frequency([]) == {}

    def test_counts_correctly(self) -> None:
        reports = [
            {"alerts": ["high_load", "voltage_drop"]},
            {"alerts": ["high_load"]},
        ]
        freq = alert_frequency(reports)
        assert freq["high_load"] == 2
        assert freq["voltage_drop"] == 1

    def test_sorted_descending(self) -> None:
        reports = [{"alerts": ["a", "b", "b", "b"]}]
        freq = alert_frequency(reports)
        keys = list(freq.keys())
        assert keys[0] == "b"


class TestComputeAnomalyRate:
    def test_zero_hours_returns_zero(self) -> None:
        assert compute_anomaly_rate(10, 0) == pytest.approx(0.0)

    def test_negative_hours_returns_zero(self) -> None:
        assert compute_anomaly_rate(5, -1) == pytest.approx(0.0)

    def test_correct_rate(self) -> None:
        assert compute_anomaly_rate(12, 4) == pytest.approx(3.0)
