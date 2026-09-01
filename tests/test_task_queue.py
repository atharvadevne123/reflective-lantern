"""Tests for app.task_queue."""

import threading

import pytest

from app.task_queue import Task, TaskQueue


class TestTask:
    def test_run_calls_fn(self):
        results = []
        t = Task(priority=1, fn=results.append, args=(42,))
        t.run()
        assert results == [42]

    def test_priority_ordering(self):
        t1 = Task(priority=10, fn=lambda: None)
        t2 = Task(priority=1, fn=lambda: None)
        assert t2 < t1


class TestTaskQueue:
    def test_submit_increases_len(self):
        q = TaskQueue(workers=0)
        q.submit(lambda: None, priority=1)
        assert len(q) == 1

    def test_workers_execute_tasks(self):
        results = []
        lock = threading.Lock()

        def work(val):
            with lock:
                results.append(val)

        q = TaskQueue(workers=2)
        q.start()
        for i in range(5):
            q.submit(work, 5, i)
        q.stop(timeout=2.0)
        assert sorted(results) == list(range(5))
        assert q.completed == 5

    def test_error_captured(self):
        def boom():
            raise ValueError("oops")

        q = TaskQueue(workers=1)
        q.start()
        q.submit(boom, 1)
        q.stop(timeout=2.0)
        assert len(q.errors) == 1
        assert isinstance(q.errors[0], ValueError)

    def test_priority_order_respected(self):
        order = []
        lock = threading.Lock()
        threading.Barrier(2)

        # Use 1 worker so tasks run sequentially
        q = TaskQueue(workers=1)

        def record(val):
            with lock:
                order.append(val)

        # Submit low-priority first, then high-priority before starting
        q.submit(record, 10, "low")
        q.submit(record, 1, "high")
        q.start()
        q.stop(timeout=2.0)
        assert order[0] == "high"
        assert order[1] == "low"

    def test_empty_queue_has_zero_length(self):
        q = TaskQueue(workers=0)
        assert len(q) == 0

    def test_multiple_errors_all_captured(self):
        def boom():
            raise RuntimeError("fail")

        q = TaskQueue(workers=1)
        q.start()
        for _ in range(3):
            q.submit(boom, 1)
        q.stop(timeout=2.0)
        assert len(q.errors) == 3

    @pytest.mark.parametrize("priority", [1, 5, 10])
    def test_task_with_varied_priorities_all_execute(self, priority: int) -> None:
        done: list[int] = []
        lock = threading.Lock()

        def record(p: int) -> None:
            with lock:
                done.append(p)

        q = TaskQueue(workers=1)
        q.start()
        q.submit(record, priority, priority)
        q.stop(timeout=2.0)
        assert priority in done
