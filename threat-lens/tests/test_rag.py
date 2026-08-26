"""Tests for threat intelligence ingestion, indexing, and retrieval."""

import json

import pytest

from app.rag_retriever import THREAT_INTEL_CORPUS, ThreatIntelRetriever
from rag.index import build_tfidf_index, load_index, save_index
from rag.ingest import load_corpus, validate_corpus
from rag.retriever import cosine_search


@pytest.fixture()
def retriever() -> ThreatIntelRetriever:
    r = ThreatIntelRetriever()
    r.build_index()
    return r


def test_search_returns_requested_count(retriever: ThreatIntelRetriever) -> None:
    assert len(retriever.search("denial of service flooding", top_k=3)) == 3


def test_search_ranks_relevant_entry_first(retriever: ThreatIntelRetriever) -> None:
    results = retriever.search("port scanning enumerate services", top_k=3)
    assert results[0]["id"] in {"MITRE-T1046", "MITRE-T1595"}


def test_search_results_sorted_by_similarity(retriever: ThreatIntelRetriever) -> None:
    sims = [r["similarity"] for r in retriever.search("brute force login", top_k=5)]
    assert sims == sorted(sims, reverse=True)


def test_search_builds_index_lazily() -> None:
    """Searching before an explicit build must still work."""
    assert ThreatIntelRetriever().search("log4j", top_k=1)


def test_corpus_entries_well_formed() -> None:
    for doc in THREAT_INTEL_CORPUS:
        assert doc["id"]
        assert len(doc["text"]) > 20


def test_validate_corpus_filters_malformed() -> None:
    docs = [
        {"id": "A", "text": "valid entry"},
        {"id": "", "text": "missing id"},
        {"id": "C"},
        "not a dict",
        {"id": "D", "text": "another valid"},
    ]
    valid = validate_corpus(docs)
    assert [d["id"] for d in valid] == ["A", "D"]


def test_load_corpus_missing_file_returns_empty(tmp_path) -> None:
    assert load_corpus(tmp_path / "nope.json") == []


def test_load_corpus_reads_file(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps([{"id": "X", "text": "some threat"}]))
    assert load_corpus(path)[0]["id"] == "X"


def test_build_index_shape() -> None:
    docs = [{"id": str(i), "text": f"threat number {i} scanning"} for i in range(4)]
    index = build_tfidf_index(docs)
    assert index["matrix"].shape[0] == 4
    assert index["ids"] == ["0", "1", "2", "3"]


def test_index_roundtrip(tmp_path) -> None:
    docs = [{"id": "a", "text": "denial of service"}, {"id": "b", "text": "port scan"}]
    path = tmp_path / "idx.npz"
    save_index(build_tfidf_index(docs), path)
    loaded = load_index(path)
    assert loaded is not None
    assert loaded["ids"] == ["a", "b"]


def test_load_index_missing_returns_none(tmp_path) -> None:
    assert load_index(tmp_path / "absent.npz") is None


def test_cosine_search_finds_match() -> None:
    docs = [{"id": "a", "text": "denial of service flooding"},
            {"id": "b", "text": "privilege escalation rootkit"}]
    index = build_tfidf_index(docs)
    results = cosine_search(
        "flooding", index["matrix"], index["vocab"], index["ids"],
        [d["text"] for d in docs], top_k=1,
    )
    assert results[0]["id"] == "a"


def test_cosine_search_unknown_terms_returns_empty() -> None:
    docs = [{"id": "a", "text": "denial of service"}]
    index = build_tfidf_index(docs)
    results = cosine_search(
        "zzzz nonexistent", index["matrix"], index["vocab"], index["ids"],
        [d["text"] for d in docs], top_k=1,
    )
    assert results == []


def test_idf_weights_never_negative() -> None:
    """A term in every document must not get a negative weight.

    The naive log(n / (1 + df)) formula goes negative for ubiquitous terms,
    which makes them repel matching documents instead of being neutral.
    """
    docs = [{"id": str(i), "text": f"common shared term unique{i}"} for i in range(5)]
    index = build_tfidf_index(docs)
    assert (index["matrix"] >= 0).all()


def test_two_doc_corpus_ranks_correctly() -> None:
    """With only two documents, unique terms must still carry signal."""
    docs = [{"id": "a", "text": "denial of service flooding"},
            {"id": "b", "text": "privilege escalation rootkit"}]
    index = build_tfidf_index(docs)
    texts = [d["text"] for d in docs]
    assert cosine_search("flooding", index["matrix"], index["vocab"],
                         index["ids"], texts, top_k=1)[0]["id"] == "a"
    assert cosine_search("rootkit", index["matrix"], index["vocab"],
                         index["ids"], texts, top_k=1)[0]["id"] == "b"
