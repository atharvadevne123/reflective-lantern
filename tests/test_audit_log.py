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

    def test_record_returns_entry(self):
        log = AuditLog()
        entry = log.record("alice", "login", "/session")
        assert isinstance(entry, AuditEntry)
        assert entry.actor == "alice"

    def test_metadata_stored_correctly(self):
        log = AuditLog()
        log.record("alice", "api_call", "/predict", model="xgb", latency_ms=42.5)
        results = log.search(actor="alice")
        assert results[0].metadata["model"] == "xgb"
        assert results[0].metadata["latency_ms"] == 42.5

    def test_search_by_resource(self):
        log = AuditLog()
        log.record("alice", "read", "doc/1")
        log.record("alice", "read", "doc/2")
        results = log.search(resource="doc/1")
        assert len(results) == 1

    def test_search_by_action(self):
        log = AuditLog()
        log.record("alice", "read", "doc/1")
        log.record("alice", "write", "doc/1")
        log.record("bob", "read", "doc/2")
        results = log.search(action="read")
        assert len(results) == 2

    def test_search_since_excludes_earlier(self):
        log = AuditLog()
        past = time.time() - 1000
        log.record("x", "a", "r")
        results = log.search(since=time.time())  # future bound
        assert len(results) == 0

    def test_export_jsonl_multiple_lines(self):
        log = AuditLog()
        log.record("alice", "login", "/session")
        log.record("bob", "logout", "/session")
        jsonl = log.export_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "actor" in parsed
            assert "timestamp" in parsed

    def test_empty_log_len_is_zero(self):
        log = AuditLog()
        assert len(log) == 0

    def test_multiple_failures_searchable(self):
        log = AuditLog()
        for i in range(5):
            log.record("bot", "brute_force", f"login/{i}", outcome="failure")
        log.record("alice", "login", "session/1")
        results = log.search(outcome="failure")
        assert len(results) == 5

    @pytest.mark.parametrize("outcome", ["success", "failure", "error", "denied"])
    def test_arbitrary_outcome_values_stored(self, outcome: str) -> None:
        log = AuditLog()
        log.record("svc", "call", "endpoint", outcome=outcome)
        results = log.search(outcome=outcome)
        assert len(results) == 1
        assert results[0].outcome == outcome
