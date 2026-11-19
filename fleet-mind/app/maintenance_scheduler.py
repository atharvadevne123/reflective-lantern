"""Maintenance scheduling utilities for fleet vehicles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class MaintenanceTask:
    vehicle_id: str
    task_type: str
    due_date: date
    priority: int = 1
    estimated_hours: float = 1.0
    notes: str = ""


@dataclass
class MaintenanceWindow:
    start_date: date
    end_date: date
    capacity_hours: float
    tasks: list[MaintenanceTask] = field(default_factory=list)

    @property
    def scheduled_hours(self) -> float:
        return sum(t.estimated_hours for t in self.tasks)

    @property
    def available_hours(self) -> float:
        return max(0.0, self.capacity_hours - self.scheduled_hours)

    @property
    def utilisation(self) -> float:
        if self.capacity_hours <= 0:
            return 0.0
        return min(1.0, self.scheduled_hours / self.capacity_hours)


def next_service_date(last_service: date, interval_days: int) -> date:
    """Return the next scheduled service date."""
    return last_service + timedelta(days=interval_days)


def days_until_due(task: MaintenanceTask, today: date | None = None) -> int:
    """Return days remaining until a task is due (negative = overdue)."""
    today = today or date.today()
    return (task.due_date - today).days


def overdue_tasks(tasks: list[MaintenanceTask], today: date | None = None) -> list[MaintenanceTask]:
    """Return tasks that are past their due date."""
    today = today or date.today()
    return [t for t in tasks if t.due_date < today]


def schedule_tasks(
    tasks: list[MaintenanceTask],
    window: MaintenanceWindow,
) -> tuple[list[MaintenanceTask], list[MaintenanceTask]]:
    """Assign tasks to a window by priority; return (scheduled, deferred)."""
    sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, t.due_date))
    scheduled: list[MaintenanceTask] = []
    deferred: list[MaintenanceTask] = []
    for task in sorted_tasks:
        if window.available_hours >= task.estimated_hours:
            window.tasks.append(task)
            scheduled.append(task)
        else:
            deferred.append(task)
    return scheduled, deferred


def utilisation_forecast(windows: list[MaintenanceWindow]) -> list[float]:
    """Return utilisation fraction for each window."""
    return [w.utilisation for w in windows]
