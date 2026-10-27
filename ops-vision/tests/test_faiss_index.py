"""Tests for ops-vision app/faiss_index.py."""

from __future__ import annotations

import pytest

from app.faiss_index import Runbook, RunbookIndex


class TestRunbook:
    def test_runbook_creation(self) -> None:
        rb = Runbook(title="High CPU", content="Reduce load", category="cpu")
        assert rb.title == "High CPU"
        assert rb.content == "Reduce load"
        assert rb.category == "cpu"


class TestRunbookIndex:
    def test_init(self) -> None:
        index = RunbookIndex(embedding_dim=16)
        assert index.embedding_dim == 16
        assert len(index.runbooks) == 0

    def test_build_with_runbooks(self) -> None:
        index = RunbookIndex(embedding_dim=16)
        runbooks = [
            Runbook("CPU Alert", "Reduce CPU usage", "cpu"),
            Runbook("Memory Alert", "Free memory", "memory"),
            Runbook("Network Issue", "Check bandwidth", "network"),
        ]
        index.build(runbooks)
        assert len(index.runbooks) == 3

    def test_search_returns_results(self) -> None:
        index = RunbookIndex(embedding_dim=16)
        runbooks = [
            Runbook("CPU Alert", "Reduce CPU usage overloaded", "cpu"),
            Runbook("Memory OOM", "Free memory pages", "memory"),
            Runbook("Disk Full", "Remove old files", "disk"),
        ]
        index.build(runbooks)
        results = index.search("CPU overloaded", top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_search_returns_runbook_objects(self) -> None:
        index = RunbookIndex(embedding_dim=16)
        runbooks = [
            Runbook("Test RB", "Test content", "test"),
        ]
        index.build(runbooks)
        results = index.search("test", top_k=1)
        if results:
            assert isinstance(results[0], Runbook)

    def test_empty_index_search(self) -> None:
        index = RunbookIndex(embedding_dim=16)
        results = index.search("anything", top_k=3)
        assert results == []

    @pytest.mark.parametrize("top_k", [1, 2, 5])
    def test_top_k_limit(self, top_k: int) -> None:
        index = RunbookIndex(embedding_dim=16)
        for i in range(3):
            index.build([Runbook(f"RB{i}", f"content {i}", "cat")])
        index.build([Runbook(f"RB{i}", f"content {i}", "cat") for i in range(3)])
        results = index.search("content", top_k=top_k)
        assert len(results) <= top_k
