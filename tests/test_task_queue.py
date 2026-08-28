"""Tests for app.task_queue."""

import threading

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
        barrier = threading.Barrier(2)

        # Use 1 worker so tasks run sequentially
        q = TaskQueue(workers=1)

        def record(val):
            with lock:
                order.append(val)

        # Submit low-priority first, then high-priority before starting
        q.submit(record, priority=10, *["low"])
        q.submit(record, priority=1, *["high"])
        q.start()
        q.stop(timeout=2.0)
        assert order[0] == "high"
        assert order[1] == "low"
