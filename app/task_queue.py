"""Priority task queue with worker-thread execution.

Provides a simple in-process task queue backed by Python's ``heapq``
module. Tasks are consumed by a configurable number of daemon threads.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Task",
    "TaskQueue",
]

logger = logging.getLogger(__name__)


@dataclass(order=True)
class Task:
    """A unit of work with a numeric priority (lower = higher priority)."""

    priority: int
    fn: Callable[..., Any] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    created_at: float = field(default_factory=time.monotonic, compare=False)

    def run(self) -> Any:
        """Execute the task and return its result."""
        return self.fn(*self.args, **self.kwargs)


class TaskQueue:
    """Thread-safe priority queue that executes tasks on worker threads."""

    def __init__(self, workers: int = 2) -> None:
        self._heap: list[Task] = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._running = False
        self._threads: list[threading.Thread] = []
        self._workers = workers
        self._completed = 0
        self._errors: list[Exception] = []

    def submit(self, fn: Callable[..., Any], priority: int = 5, *args: Any, **kwargs: Any) -> None:
        """Enqueue a task; lower *priority* values run first."""
        task = Task(priority=priority, fn=fn, args=args, kwargs=kwargs)
        with self._not_empty:
            heapq.heappush(self._heap, task)
            self._not_empty.notify()

    def start(self) -> None:
        """Start worker threads."""
        self._running = True
        for _ in range(self._workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal workers to stop and wait for them."""
        with self._not_empty:
            self._running = False
            self._not_empty.notify_all()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()

    def _worker(self) -> None:
        while True:
            with self._not_empty:
                while not self._heap and self._running:
                    self._not_empty.wait(timeout=0.1)
                if not self._heap:
                    return
                task = heapq.heappop(self._heap)
            try:
                task.run()
                with self._lock:
                    self._completed += 1
            except Exception as exc:
                logger.exception("Task failed: %s", exc)
                with self._lock:
                    self._errors.append(exc)

    @property
    def completed(self) -> int:
        """Return the number of tasks that have finished successfully."""
        with self._lock:
            return self._completed

    @property
    def errors(self) -> list[Exception]:
        """Return a snapshot of all exceptions raised by failed tasks."""
        with self._lock:
            return list(self._errors)

    def __len__(self) -> int:
        """Return the number of tasks currently waiting in the queue."""
        with self._lock:
            return len(self._heap)
