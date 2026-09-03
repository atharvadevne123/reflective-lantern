"""Tests for app.audit_log."""

import json
import time
import pytest

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


class TestAuditLogExtensions:
    def test_clear_removes_all_entries(self):
        from app.audit_log import AuditLog
        log = AuditLog()
        log.record("alice", "create", "doc/1")
        log.record("bob", "delete", "doc/2")
        log.clear()
        assert len(log) == 0

    def test_count_all_entries(self):
        from app.audit_log import AuditLog
        log = AuditLog()
        for i in range(5):
            log.record("u", "read", f"file/{i}")
        assert log.count() == 5

    def test_count_by_actor(self):
        from app.audit_log import AuditLog
        log = AuditLog()
        log.record("alice", "read", "x")
        log.record("alice", "write", "y")
        log.record("bob", "read", "z")
        assert log.count(actor="alice") == 2
        assert log.count(actor="bob") == 1

    def test_actors_returns_sorted_unique(self):
        from app.audit_log import AuditLog
        log = AuditLog()
        log.record("charlie", "x", "r")
        log.record("alice", "x", "r")
        log.record("alice", "y", "r")
        log.record("bob", "x", "r")
        assert log.actors() == ["alice", "bob", "charlie"]

    @pytest.mark.parametrize("outcome", ["success", "failure", "error"])
    def test_search_by_outcome(self, outcome: str):
        from app.audit_log import AuditLog
        log = AuditLog()
        log.record("u", "act", "r", outcome=outcome)
        log.record("u", "act", "r", outcome="other")
        results = log.search(outcome=outcome)
        assert len(results) == 1
        assert results[0].outcome == outcome
