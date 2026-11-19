"""Tests for fleet-mind app/maintenance_scheduler.py."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.maintenance_scheduler import (
    MaintenanceTask,
    MaintenanceWindow,
    days_until_due,
    next_service_date,
    overdue_tasks,
    schedule_tasks,
    utilisation_forecast,
)


class TestNextServiceDate:
    def test_adds_interval(self) -> None:
        last = date(2026, 1, 1)
        assert next_service_date(last, 90) == date(2026, 4, 1)

    def test_zero_interval_returns_same(self) -> None:
        last = date(2026, 6, 1)
        assert next_service_date(last, 0) == last


class TestDaysUntilDue:
    def test_future_task_positive(self) -> None:
        today = date(2026, 11, 19)
        task = MaintenanceTask("v1", "oil_change", date(2026, 12, 19))
        assert days_until_due(task, today) == 30

    def test_overdue_task_negative(self) -> None:
        today = date(2026, 11, 19)
        task = MaintenanceTask("v1", "oil_change", date(2026, 11, 9))
        assert days_until_due(task, today) == -10


class TestOverdueTasks:
    def test_returns_past_due_tasks(self) -> None:
        today = date(2026, 11, 19)
        past = MaintenanceTask("v1", "t", date(2026, 11, 1))
        future = MaintenanceTask("v2", "t", date(2026, 12, 1))
        result = overdue_tasks([past, future], today)
        assert len(result) == 1
        assert result[0].vehicle_id == "v1"

    def test_empty_list(self) -> None:
        assert overdue_tasks([], date(2026, 11, 19)) == []


class TestScheduleTasks:
    def test_fits_within_capacity(self) -> None:
        window = MaintenanceWindow(date(2026, 12, 1), date(2026, 12, 7), 10.0)
        tasks = [MaintenanceTask("v1", "t", date(2026, 12, 2), estimated_hours=3.0)]
        scheduled, deferred = schedule_tasks(tasks, window)
        assert len(scheduled) == 1
        assert len(deferred) == 0

    def test_defers_when_over_capacity(self) -> None:
        window = MaintenanceWindow(date(2026, 12, 1), date(2026, 12, 7), 2.0)
        tasks = [
            MaintenanceTask("v1", "t", date(2026, 12, 2), estimated_hours=3.0),
        ]
        scheduled, deferred = schedule_tasks(tasks, window)
        assert len(scheduled) == 0
        assert len(deferred) == 1


class TestMaintenanceWindow:
    def test_scheduled_hours(self) -> None:
        window = MaintenanceWindow(date(2026, 12, 1), date(2026, 12, 7), 10.0)
        window.tasks = [
            MaintenanceTask("v1", "t", date(2026, 12, 2), estimated_hours=3.0),
            MaintenanceTask("v2", "t", date(2026, 12, 3), estimated_hours=2.0),
        ]
        assert window.scheduled_hours == pytest.approx(5.0)

    def test_utilisation_clamped(self) -> None:
        window = MaintenanceWindow(date(2026, 12, 1), date(2026, 12, 7), 0.0)
        assert window.utilisation == pytest.approx(0.0)


class TestUtilisationForecast:
    def test_returns_per_window_fractions(self) -> None:
        w1 = MaintenanceWindow(date(2026, 12, 1), date(2026, 12, 7), 10.0)
        w2 = MaintenanceWindow(date(2026, 12, 8), date(2026, 12, 14), 10.0)
        w1.tasks = [MaintenanceTask("v1", "t", date(2026, 12, 2), estimated_hours=5.0)]
        fracs = utilisation_forecast([w1, w2])
        assert fracs[0] == pytest.approx(0.5)
        assert fracs[1] == pytest.approx(0.0)
