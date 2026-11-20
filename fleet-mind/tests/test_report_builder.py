"""Tests for fleet-mind app/report_builder.py."""

from __future__ import annotations

from datetime import date

from app.report_builder import (
    FleetReport,
    ReportSection,
    build_maintenance_report,
    filter_rows,
    report_to_dict,
)


class TestReportSection:
    def test_default_rows_empty(self) -> None:
        section = ReportSection("Title")
        assert section.rows == []


class TestFleetReport:
    def test_add_section(self) -> None:
        report = FleetReport("Test Report", date(2026, 11, 19))
        section = ReportSection("Summary", rows=[{"id": 1}])
        report.add_section(section)
        assert len(report.sections) == 1

    def test_row_count(self) -> None:
        report = FleetReport("Test", date(2026, 11, 19))
        report.add_section(ReportSection("A", rows=[{}, {}]))
        report.add_section(ReportSection("B", rows=[{}]))
        assert report.row_count() == 3


class TestBuildMaintenanceReport:
    def test_has_two_sections(self) -> None:
        report = build_maintenance_report(
            vehicle_summaries=[{"id": "v1"}],
            overdue_tasks=[],
        )
        assert len(report.sections) == 2

    def test_title_set(self) -> None:
        report = build_maintenance_report([], [])
        assert "Maintenance" in report.title


class TestReportToDict:
    def test_contains_title(self) -> None:
        report = FleetReport("Fleet Report", date(2026, 11, 19))
        d = report_to_dict(report)
        assert d["title"] == "Fleet Report"

    def test_generated_on_iso_format(self) -> None:
        report = FleetReport("R", date(2026, 11, 19))
        d = report_to_dict(report)
        assert d["generated_on"] == "2026-11-19"

    def test_sections_serialised(self) -> None:
        report = FleetReport("R", date(2026, 11, 19))
        report.add_section(ReportSection("S1", rows=[{"x": 1}]))
        d = report_to_dict(report)
        assert len(d["sections"]) == 1
        assert d["sections"][0]["title"] == "S1"


class TestFilterRows:
    def test_filters_matching_rows(self) -> None:
        section = ReportSection("S", rows=[{"status": "ok"}, {"status": "fail"}])
        filtered = filter_rows(section, lambda r: r["status"] == "ok")
        assert len(filtered.rows) == 1

    def test_non_callable_returns_original(self) -> None:
        section = ReportSection("S", rows=[{"x": 1}])
        result = filter_rows(section, None)
        assert result.rows == section.rows
