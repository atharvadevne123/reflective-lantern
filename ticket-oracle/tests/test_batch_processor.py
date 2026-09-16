"""Batch processor tests for Ticket-Oracle."""

from __future__ import annotations

import pytest

from app.batch_processor import aggregate_batch_stats, process_batch


def _ok_predict(ticket):
    return {
        "ticket_id": ticket["ticket_id"],
        "priority": "P2",
        "sla_breach_risk": 0.3,
    }


def _fail_predict(ticket):
    raise RuntimeError("inference error")


def test_process_batch_all_success():
    tickets = [{"ticket_id": f"T-{i}"} for i in range(5)]
    result = process_batch(tickets, _ok_predict)
    assert len(result["results"]) == 5
    assert len(result["errors"]) == 0


def test_process_batch_all_fail():
    tickets = [{"ticket_id": f"T-{i}"} for i in range(3)]
    result = process_batch(tickets, _fail_predict)
    assert len(result["results"]) == 0
    assert len(result["errors"]) == 3


def test_process_batch_mixed():
    def mixed_predict(ticket):
        if ticket["ticket_id"] == "T-1":
            raise ValueError("bad")
        return {"ticket_id": ticket["ticket_id"], "priority": "P3", "sla_breach_risk": 0.1}

    tickets = [{"ticket_id": f"T-{i}"} for i in range(4)]
    result = process_batch(tickets, mixed_predict)
    assert len(result["results"]) == 3
    assert len(result["errors"]) == 1
    assert result["errors"][0]["ticket_id"] == "T-1"


def test_process_batch_chunked():
    tickets = [{"ticket_id": f"T-{i}"} for i in range(10)]
    result = process_batch(tickets, _ok_predict, chunk_size=3)
    assert len(result["results"]) == 10


def test_process_batch_empty():
    result = process_batch([], _ok_predict)
    assert result["results"] == []
    assert result["errors"] == []


def test_aggregate_stats_empty():
    stats = aggregate_batch_stats([])
    assert stats["total"] == 0
    assert stats["avg_sla_breach_risk"] == 0.0


def test_aggregate_stats_counts():
    results = [
        {"priority": "P1", "sla_breach_risk": 0.9},
        {"priority": "P2", "sla_breach_risk": 0.4},
        {"priority": "P2", "sla_breach_risk": 0.2},
    ]
    stats = aggregate_batch_stats(results)
    assert stats["priority_counts"]["P1"] == 1
    assert stats["priority_counts"]["P2"] == 2
    assert stats["avg_sla_breach_risk"] == pytest.approx(0.5, abs=0.01)


def test_aggregate_stats_total():
    results = [{"priority": "P4", "sla_breach_risk": 0.05}] * 7
    stats = aggregate_batch_stats(results)
    assert stats["total"] == 7
