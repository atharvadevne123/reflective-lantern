"""FAISS-based ticket similarity search for Ticket-Oracle.

Indexes historical ticket feature vectors so new tickets can be matched
to the most similar past tickets for context-aware escalation hints.
Falls back to brute-force cosine search when FAISS is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path("models/ticket_index.faiss")
_index = None
_metadata: list[dict[str, Any]] = []


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def build_index(
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    persist: bool = False,
) -> None:
    """Build an L2 FAISS index from dense ticket feature vectors.

    Args:
        vectors: (n, d) float32 array of feature vectors.
        metadata: List of per-ticket metadata dicts (same order as vectors).
        persist: If True, save the index to FAISS_INDEX_PATH.
    """
    global _index, _metadata

    vectors = vectors.astype(np.float32)
    faiss = _try_import_faiss()

    if faiss is not None:
        idx = faiss.IndexFlatL2(vectors.shape[1])
        idx.add(vectors)
        _index = idx
        if persist:
            FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(idx, str(FAISS_INDEX_PATH))
    else:
        _index = vectors

    _metadata = list(metadata)
    logger.info("similarity_index_built", extra={"n": len(metadata), "backend": "faiss" if faiss else "brute"})


def search_similar(query: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k most similar historical tickets to a query vector.

    Args:
        query: (d,) or (1, d) float32 feature vector.
        top_k: Number of nearest neighbours to return.

    Returns:
        List of metadata dicts for the top-k matches, each annotated with
        a 'distance' key.

    Raises:
        RuntimeError: If the index has not been built yet.
    """
    if _index is None:
        raise RuntimeError("Similarity index not built. Call build_index() first.")

    q = query.reshape(1, -1).astype(np.float32)
    top_k = min(top_k, len(_metadata))
    faiss = _try_import_faiss()

    if faiss is not None and hasattr(_index, "search"):
        distances, indices = _index.search(q, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(_metadata):
                continue
            results.append({**_metadata[idx], "distance": round(float(dist), 4)})
        return results
    else:
        # Brute-force L2 over stored vectors
        stored = _index
        diffs = stored - q
        dists = np.sum(diffs ** 2, axis=1)
        top_indices = np.argsort(dists)[:top_k]
        return [{**_metadata[i], "distance": round(float(dists[i]), 4)} for i in top_indices]


def is_index_built() -> bool:
    """Return True if the similarity index has been initialised."""
    return _index is not None
