"""Tests for fleet summary and drift report generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.reporting import fleet_summary, recent_drift_reports


class TestFleetSummary:
    def test_returns_dict(self, db_session):
        result = fleet_summary(db_session)
        assert isinstance(result, dict)

    def test_has_total_predictions(self, db_session):
        result = fleet_summary(db_session)
        assert "total_predictions" in result

    def test_maintenance_rate_between_zero_one(self, db_session):
        result = fleet_summary(db_session)
        rate = result.get("maintenance_rate", 0.0)
        assert 0.0 <= rate <= 1.0

    def test_top_anomaly_types_is_list(self, db_session):
        result = fleet_summary(db_session)
        anomalies = result.get("top_anomaly_types", [])
        assert isinstance(anomalies, list)

    @pytest.mark.parametrize("limit", [10, 50, 200])
    def test_respects_limit(self, db_session, limit):
        result = fleet_summary(db_session, limit=limit)
        assert result.get("total_predictions", 0) <= limit


class TestRecentDriftReports:
    def test_returns_list(self, db_session):
        result = recent_drift_reports(db_session)
        assert isinstance(result, list)

    def test_report_entries_have_required_keys(self, db_session):
        results = recent_drift_reports(db_session)
        for entry in results:
            assert "feature" in entry
            assert "drift_detected" in entry

    @pytest.mark.parametrize("limit", [5, 10, 50])
    def test_respects_limit(self, db_session, limit):
        results = recent_drift_reports(db_session, limit=limit)
        assert len(results) <= limit
