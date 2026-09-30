"""FAISS-based historical match similarity search for Sport-Cast."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path("match_index.faiss")
_INDEX = None
_INDEX_DIM: int = 0


def build_index(vectors: np.ndarray, *, persist: bool = False) -> object:
    """Build a FAISS flat L2 index from match feature vectors.

    Args:
        vectors: 2D array of shape (n_matches, n_features).
        persist: If True, save index to FAISS_INDEX_PATH.

    Returns:
        A FAISS IndexFlatL2 index.
    """
    global _INDEX, _INDEX_DIM
    try:
        import faiss

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors.astype(np.float32))
        _INDEX = index
        _INDEX_DIM = dim
        if persist:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
        logger.info("FAISS index built: %d vectors, dim=%d", len(vectors), dim)
        return index
    except ImportError:
        logger.warning("faiss-cpu not installed; using brute-force fallback")
        return _BruteForceIndex(vectors)


def search_similar_matches(
    query: np.ndarray,
    top_k: int = 5,
    *,
    index: object | None = None,
) -> list[dict]:
    """Find top-k most similar historical match feature vectors.

    Args:
        query: 1D feature vector for the query match.
        top_k: Number of nearest neighbours to return.
        index: Optional pre-built FAISS index; uses module-level index if None.

    Returns:
        List of dicts with keys 'rank', 'distance', 'index'.
    """
    idx = index or _INDEX
    if idx is None:
        return []
    query_2d = query.reshape(1, -1).astype(np.float32)
    if isinstance(idx, _BruteForceIndex):
        return idx.search(query_2d, top_k)
    try:
        distances, indices = idx.search(query_2d, top_k)
        return [
            {"rank": i + 1, "distance": float(distances[0][i]), "index": int(indices[0][i])}
            for i in range(top_k)
            if indices[0][i] >= 0
        ]
    except (AttributeError, ValueError):
        return idx.search(query_2d, top_k)


class _BruteForceIndex:
    """Brute-force L2 search fallback when faiss-cpu is not installed."""

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors.astype(np.float32)

    def search(self, query: np.ndarray, top_k: int) -> list[dict]:
        diffs = self._vectors - query
        distances = (diffs ** 2).sum(axis=1)
        indices = np.argsort(distances)[:top_k]
        return [
            {"rank": i + 1, "distance": float(distances[indices[i]]), "index": int(indices[i])}
            for i in range(min(top_k, len(indices)))
        ]
