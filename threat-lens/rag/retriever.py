"""Query interface for the threat intelligence vector index."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def cosine_search(
    query: str,
    matrix: np.ndarray,
    vocab: dict[str, int],
    ids: list[str],
    texts: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve top-k documents by cosine similarity to the query.

    Args:
        query: Free-text query string.
        matrix: TF-IDF document matrix of shape (n_docs, vocab_size).
        vocab: Vocabulary mapping word → column index.
        ids: Document identifiers aligned with matrix rows.
        texts: Raw document texts aligned with matrix rows.
        top_k: Number of results to return.

    Returns:
        Ranked list of {id, text, similarity} dicts.
    """
    q_vec = np.zeros(len(vocab), dtype=np.float32)
    for word in query.lower().split():
        if word in vocab:
            q_vec[vocab[word]] += 1.0

    norm = float(np.linalg.norm(q_vec))
    if norm < 1e-9:
        logger.debug("Zero-norm query vector; returning empty results")
        return []
    q_vec /= norm

    row_norms = np.linalg.norm(matrix, axis=1) + 1e-9
    sims = (matrix @ q_vec) / row_norms
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "id": ids[i],
            "text": texts[i],
            "similarity": round(float(sims[i]), 4),
        }
        for i in top_idx
    ]
