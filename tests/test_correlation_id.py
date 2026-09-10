"""Tests for app.correlation_id."""

import threading

from app.correlation_id import (
    clear_correlation_id,
    correlation_context,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
)


class TestNewCorrelationId:
    def test_is_string(self):
        assert isinstance(new_correlation_id(), str)

    def test_unique(self):
        ids = {new_correlation_id() for _ in range(100)}
        assert len(ids) == 100


class TestGetSet:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_none_by_default(self):
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"

    def test_clear(self):
        set_correlation_id("xyz")
        clear_correlation_id()
        assert get_correlation_id() is None


class TestCorrelationContext:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_yields_cid(self):
        with correlation_context("req-1") as cid:
            assert cid == "req-1"
            assert get_correlation_id() == "req-1"

    def test_restores_none_after(self):
        with correlation_context("req-2"):
            pass
        assert get_correlation_id() is None

    def test_restores_previous_cid(self):
        set_correlation_id("outer")
        with correlation_context("inner") as cid:
            assert cid == "inner"
        assert get_correlation_id() == "outer"

    def test_auto_generates_cid(self):
        with correlation_context() as cid:
            assert len(cid) == 36  # UUID4 format

    def test_restores_on_exception(self):
        set_correlation_id("before")
        try:
            with correlation_context("during"):
                raise RuntimeError
        except RuntimeError:
            pass
        assert get_correlation_id() == "before"


class TestThreadIsolation:
    def test_threads_have_independent_ids(self):
        results = {}

        def worker(name, cid):
            with correlation_context(cid):
                import time

                time.sleep(0.01)
                results[name] = get_correlation_id()

        threads = [threading.Thread(target=worker, args=(f"t{i}", f"id-{i}")) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(5):
            assert results[f"t{i}"] == f"id-{i}"

    def test_main_thread_unaffected_by_worker(self):
        set_correlation_id("main-cid")
        results = []

        def worker():
            set_correlation_id("worker-cid")
            results.append(get_correlation_id())

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert get_correlation_id() == "main-cid"
        assert results == ["worker-cid"]
        clear_correlation_id()


class TestNewCorrelationIdFormat:
    def test_uuid_format(self):
        import re

        cid = new_correlation_id()
        assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", cid)

    def test_length_36_chars(self):
        assert len(new_correlation_id()) == 36

    def test_contains_hyphens(self):
        cid = new_correlation_id()
        assert cid.count("-") == 4

    def test_generated_ids_all_unique_in_bulk(self):
        ids = [new_correlation_id() for _ in range(500)]
        assert len(set(ids)) == 500


class TestSetCorrelationIdEdgeCases:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_set_empty_string(self):
        set_correlation_id("")
        assert get_correlation_id() == ""

    def test_set_overwrite_existing(self):
        set_correlation_id("first")
        set_correlation_id("second")
        assert get_correlation_id() == "second"
