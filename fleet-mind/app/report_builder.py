"""Build structured maintenance and fleet health reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class ReportSection:
    title: str
    rows: list[dict[str, object]] = field(default_factory=list)
    summary: str = ""


@dataclass
class FleetReport:
    title: str
    generated_on: date
    sections: list[ReportSection] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        self.sections.append(section)

    def row_count(self) -> int:
        return sum(len(s.rows) for s in self.sections)


def build_maintenance_report(
    vehicle_summaries: list[dict[str, object]],
    overdue_tasks: list[dict[str, object]],
    today: date | None = None,
) -> FleetReport:
    """Build a standard maintenance report."""
    today = today or date.today()
    report = FleetReport(title="Fleet Maintenance Report", generated_on=today)

    summary_section = ReportSection(
        title="Vehicle Summary",
        rows=vehicle_summaries,
        summary=f"{len(vehicle_summaries)} vehicles in fleet",
    )
    report.add_section(summary_section)

    overdue_section = ReportSection(
        title="Overdue Maintenance",
        rows=overdue_tasks,
        summary=f"{len(overdue_tasks)} tasks overdue",
    )
    report.add_section(overdue_section)
    return report


def report_to_dict(report: FleetReport) -> dict[str, object]:
    """Serialise a FleetReport to a plain dict."""
    return {
        "title": report.title,
        "generated_on": report.generated_on.isoformat(),
        "metadata": report.metadata,
        "sections": [
            {
                "title": s.title,
                "summary": s.summary,
                "rows": s.rows,
            }
            for s in report.sections
        ],
    }


def filter_rows(section: ReportSection, predicate: object) -> ReportSection:
    """Return a new section with rows that satisfy the predicate callable."""
    if not callable(predicate):
        return section
    return ReportSection(
        title=section.title,
        rows=[r for r in section.rows if predicate(r)],
        summary=section.summary,
    )
