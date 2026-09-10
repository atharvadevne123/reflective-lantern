"""Cosine-similarity retriever over the neighbourhood RAG index.

Queries are embedded with the same TF-IDF vectorizer used at index time,
then ranked by cosine similarity against the stored document vectors.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from rag.index import build_index, load_index
from rag.ingest import load_documents

logger = logging.getLogger(__name__)

_vectorizer = None
_vectors: np.ndarray | None = None
_doc_ids: list[str] | None = None
_texts: dict[str, str] = {}


def _ensure_loaded() -> None:
    """Lazy-load the index and document texts into module-level cache."""
    global _vectorizer, _vectors, _doc_ids, _texts
    if _vectorizer is None:
        build_index()
        _vectorizer, _vectors, _doc_ids = load_index()
        docs = load_documents()
        _texts = {d["id"]: d["text"] for d in docs}
        logger.info("RAG retriever initialised — %d documents", len(_doc_ids))


def retrieve(query: str, top_k: int = 2) -> list[dict[str, object]]:
    """Retrieve the top-k most relevant neighbourhood documents for a query.

    Args:
        query: Free-text query string (e.g. "high rental yield urban property").
        top_k: Number of documents to return.

    Returns:
        List of dicts with keys: id, text, score.
    """
    _ensure_loaded()
    q_vec = _vectorizer.transform([query]).toarray().astype(float)
    scores = cosine_similarity(q_vec, _vectors)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc_id = _doc_ids[idx]
        results.append({
            "id": doc_id,
            "text": _texts.get(doc_id, ""),
            "score": round(float(scores[idx]), 4),
        })
        logger.debug("RAG hit — id=%s score=%.4f", doc_id, scores[idx])
    return results


def neighbourhood_summary(neighborhood: str) -> str:
    """Return the market report text for a specific neighbourhood.

    Args:
        neighborhood: Exact neighbourhood identifier string.

    Returns:
        Market report paragraph, or a generic message if not found.
    """
    _ensure_loaded()
    return _texts.get(neighborhood, f"No market report available for '{neighborhood}'.")
