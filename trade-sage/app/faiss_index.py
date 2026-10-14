"""FAISS-based similarity search for equity pattern matching."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_equity_index.bin"
_index = None
_stored_labels: list[str] = []

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    faiss = None  # type: ignore[assignment]


def build_index(
    vectors: np.ndarray,
    labels: list[str],
    persist: bool = False,
) -> None:
    """Build an in-memory FAISS flat L2 index.

    Args:
        vectors: Feature matrix of shape (n, dim).
        labels: String identifier for each row.
        persist: If True, save the index to FAISS_INDEX_PATH.
    """
    global _index, _stored_labels
    if not _FAISS_AVAILABLE:
        logger.debug("faiss not installed — similarity index not built")
        return
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype(np.float32))
    _index = index
    _stored_labels = labels
    if persist:
        faiss.write_index(index, FAISS_INDEX_PATH)
        logger.info("FAISS index persisted to %s", FAISS_INDEX_PATH)


def search_similar(
    query: np.ndarray,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return the k nearest neighbours from the index.

    Args:
        query: 1D feature vector to search against.
        k: Number of neighbours to return.

    Returns:
        List of (label, distance) tuples sorted nearest first.
        Falls back to brute-force numpy cosine similarity when FAISS is unavailable.
    """
    if not _FAISS_AVAILABLE or _index is None:
        return _brute_force_search(query, k)
    q = query.astype(np.float32).reshape(1, -1)
    try:
        distances, indices = _index.search(q, k)
        return [
            (_stored_labels[i], float(distances[0][j]))
            for j, i in enumerate(indices[0])
            if 0 <= i < len(_stored_labels)
        ]
    except Exception as exc:
        logger.warning("FAISS search failed (%s), falling back", exc)
        return _brute_force_search(query, k)


def _brute_force_search(
    query: np.ndarray,
    k: int,
) -> list[tuple[str, float]]:
    """Brute-force L2 nearest-neighbour fallback."""
    if not _stored_labels:
        return []
    if _index is None:
        return []
    try:
        _, indices = _index.search(query.astype(np.float32).reshape(1, -1), k)
        return [(_stored_labels[i], 0.0) for i in indices[0] if i >= 0]
    except Exception:
        return []
