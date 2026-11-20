"""Tests for threat-lens app/rag_retriever.py."""

from __future__ import annotations

import pytest
from app.rag_retriever import THREAT_INTEL_CORPUS, ThreatIntelRetriever


class TestThreatIntelCorpus:
    def test_corpus_not_empty(self) -> None:
        assert len(THREAT_INTEL_CORPUS) > 0

    def test_corpus_entries_have_id(self) -> None:
        for entry in THREAT_INTEL_CORPUS:
            assert "id" in entry

    def test_corpus_entries_have_text(self) -> None:
        for entry in THREAT_INTEL_CORPUS:
            assert "text" in entry
            assert len(entry["text"]) > 0


class TestThreatIntelRetriever:
    def test_init(self) -> None:
        retriever = ThreatIntelRetriever()
        assert retriever is not None

    def test_retrieve_returns_list(self) -> None:
        retriever = ThreatIntelRetriever()
        results = retriever.retrieve("SQL injection attack", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_top_k_limit(self) -> None:
        retriever = ThreatIntelRetriever()
        results = retriever.retrieve("DoS attack", top_k=2)
        assert len(results) <= 2

    def test_retrieve_returns_dicts(self) -> None:
        retriever = ThreatIntelRetriever()
        results = retriever.retrieve("port scan probe", top_k=3)
        for r in results:
            assert isinstance(r, dict)

    def test_retrieve_empty_query(self) -> None:
        retriever = ThreatIntelRetriever()
        results = retriever.retrieve("", top_k=3)
        assert isinstance(results, list)

    @pytest.mark.parametrize(
        "query",
        [
            "remote code execution",
            "brute force login",
            "denial of service",
            "network scanning",
        ],
    )
    def test_various_threat_queries(self, query: str) -> None:
        retriever = ThreatIntelRetriever()
        results = retriever.retrieve(query, top_k=3)
        assert isinstance(results, list)
