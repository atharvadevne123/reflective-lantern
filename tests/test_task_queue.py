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

    def test_empty_queue_len_zero(self):
        q = TaskQueue(workers=0)
        assert len(q) == 0

    def test_errors_list_empty_on_clean_run(self):
        q = TaskQueue(workers=1)
        q.start()
        q.submit(lambda: None, 1)
        q.stop(timeout=2.0)
        assert q.errors == []

    def test_multiple_errors_all_captured(self):
        def explode():
            raise RuntimeError("boom")

        q = TaskQueue(workers=1)
        q.start()
        for _ in range(3):
            q.submit(explode, 1)
        q.stop(timeout=2.0)
        assert len(q.errors) == 3
        assert all(isinstance(e, RuntimeError) for e in q.errors)

    def test_completed_count_zero_before_start(self):
        q = TaskQueue(workers=1)
        assert q.completed == 0

    def test_stop_without_start_is_safe(self):
        q = TaskQueue(workers=2)
        q.stop(timeout=0.1)  # should not raise

    def test_task_kwargs_passed_correctly(self):
        results = {}

        def store(**kw):
            results.update(kw)

        t = Task(priority=0, fn=store, kwargs={"a": 1, "b": 2})
        t.run()
        assert results == {"a": 1, "b": 2}

    def test_high_volume_tasks_all_complete(self):
        import time

        counter = {"n": 0}
        lock = threading.Lock()

        def inc():
            with lock:
                counter["n"] += 1

        n = 50
        q = TaskQueue(workers=4)
        q.start()
        for _ in range(n):
            q.submit(inc, 1)
        deadline = time.monotonic() + 5.0
        while q.completed < n and time.monotonic() < deadline:
            time.sleep(0.05)
        q.stop(timeout=2.0)
        assert counter["n"] == n


class TestTaskQueueEdgeCases:
    def test_submit_after_stop_does_not_raise(self):
        q = TaskQueue(workers=1)
        q.start()
        q.stop(timeout=1.0)
        q.submit(lambda: None, priority=1)  # should not raise

    def test_task_created_at_is_set(self):
        import time

        before = time.monotonic()
        t = Task(priority=0, fn=lambda: None)
        after = time.monotonic()
        assert before <= t.created_at <= after

    def test_task_with_both_args_and_kwargs(self):
        results = {}

        def fn(pos, *, kw):
            results["pos"] = pos
            results["kw"] = kw

        t = Task(priority=0, fn=fn, args=(99,), kwargs={"kw": "ok"})
        t.run()
        assert results == {"pos": 99, "kw": "ok"}

    @pytest.mark.parametrize("n_workers", [1, 2, 4])
    def test_concurrent_workers_complete_all(self, n_workers):
        results = []
        lock = threading.Lock()

        def work(i):
            with lock:
                results.append(i)

        q = TaskQueue(workers=n_workers)
        q.start()
        for i in range(10):
            q.submit(work, 1, i)
        q.stop(timeout=5.0)
        assert sorted(results) == list(range(10))

    def test_zero_workers_queue_accepts_tasks(self):
        q = TaskQueue(workers=0)
        for i in range(5):
            q.submit(lambda: None, priority=i)
        assert len(q) == 5
