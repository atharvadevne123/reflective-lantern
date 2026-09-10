"""Tests for the RAG neighbourhood knowledge pipeline."""

import pytest

from rag.ingest import NEIGHBOURHOOD_DOCS, ingest_documents, load_documents
from rag.retriever import neighbourhood_summary, retrieve


def test_neighbourhood_docs_not_empty():
    assert len(NEIGHBOURHOOD_DOCS) >= 10


def test_all_docs_have_id_and_text():
    for doc in NEIGHBOURHOOD_DOCS:
        assert "id" in doc and "text" in doc
        assert len(doc["text"]) > 20


def test_ingest_writes_corpus(tmp_path):
    corpus = tmp_path / "corpus.json"
    n = ingest_documents(output_path=corpus)
    assert n == len(NEIGHBOURHOOD_DOCS)
    assert corpus.exists()


def test_load_documents_returns_list(tmp_path):
    corpus = tmp_path / "corpus.json"
    ingest_documents(output_path=corpus)
    docs = load_documents(corpus_path=corpus)
    assert isinstance(docs, list)
    assert len(docs) == len(NEIGHBOURHOOD_DOCS)


def test_retrieve_returns_top_k():
    results = retrieve("high rental yield investment", top_k=2)
    assert len(results) == 2
    for r in results:
        assert "id" in r and "text" in r and "score" in r


def test_retrieve_scores_are_bounded():
    results = retrieve("luxury waterfront property", top_k=3)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_retrieve_waterfront_ranks_high():
    results = retrieve("waterfront luxury high price premium", top_k=3)
    ids = [r["id"] for r in results]
    assert "waterfront" in ids


def test_neighbourhood_summary_known():
    text = neighbourhood_summary("suburb")
    assert "suburb" in text.lower() or len(text) > 30


def test_neighbourhood_summary_unknown():
    text = neighbourhood_summary("atlantis")
    assert "not found" in text.lower() or "No market report" in text


@pytest.mark.parametrize("neighbourhood", [
    "downtown", "waterfront", "suburb", "rural", "university",
])
def test_summary_all_known_neighbourhoods(neighbourhood):
    text = neighbourhood_summary(neighbourhood)
    assert len(text) > 20
