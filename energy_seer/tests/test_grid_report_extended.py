"""Extended tests for energy_seer app/grid_report.py."""

from __future__ import annotations

import pytest

from app.grid_report import generate_grid_report


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
