"""Tests for app.audit_log."""

import json
import time

import pytest

from app.audit_log import AuditEntry, AuditLog


class TestAuditEntry:
    def test_frozen(self):
        entry = AuditEntry(actor="alice", action="read", resource="doc/1")
        with pytest.raises((TypeError, AttributeError)):
            entry.actor = "bob"  # type: ignore[misc]

    def test_default_outcome(self):
        entry = AuditEntry(actor="a", action="b", resource="c")
        assert entry.outcome == "success"


class TestAuditLog:
    def test_len(self):
        log = AuditLog()
        log.record("alice", "login", "/session")
        log.record("bob", "logout", "/session")
        assert len(log) == 2

    def test_search_by_actor(self):
        log = AuditLog()
        log.record("alice", "read", "doc/1")
        log.record("bob", "write", "doc/2")
        results = log.search(actor="alice")
        assert len(results) == 1
        assert results[0].actor == "alice"

    def test_search_by_outcome(self):
        log = AuditLog()
        log.record("alice", "delete", "doc/3", outcome="failure")
        log.record("alice", "read", "doc/4")
        failures = log.search(outcome="failure")
        assert len(failures) == 1

    def test_search_by_time_range(self):
        log = AuditLog()
        t0 = time.time()
        log.record("x", "a", "r")
        t1 = time.time()
        results = log.search(since=t0, until=t1)
        assert len(results) == 1

    def test_search_combined_filters(self):
        log = AuditLog()
        log.record("alice", "write", "doc/1")
        log.record("alice", "read", "doc/2")
        log.record("bob", "write", "doc/3")
        results = log.search(actor="alice", action="write")
        assert len(results) == 1
        assert results[0].resource == "doc/1"

    def test_export_jsonl(self):
        log = AuditLog()
        log.record("alice", "login", "/session", ip="127.0.0.1")
        jsonl = log.export_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["actor"] == "alice"
        assert parsed["metadata"]["ip"] == "127.0.0.1"

    def test_empty_search_returns_all(self):
        log = AuditLog()
        log.record("a", "b", "c")
        log.record("d", "e", "f")
        assert len(log.search()) == 2

    import pytest

    @pytest.mark.parametrize("n", [1, 3, 10])
    def test_record_count_matches(self, n: int) -> None:
        from app.audit_log import AuditLog

        log = AuditLog()
        for i in range(n):
            log.record(f"user{i}", "read", f"res/{i}")
        assert len(log.search()) == n

    def test_record_timestamp_is_string(self) -> None:
        from app.audit_log import AuditLog

        log = AuditLog()
        log.record("alice", "read", "doc/1")
        entry = log.search()[0]
        assert isinstance(entry.timestamp, str)
