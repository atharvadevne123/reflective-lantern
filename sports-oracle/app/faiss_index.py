"""FAISS-based match similarity search for Sports-Oracle."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path("faiss_index.bin")


def build_index(embeddings: np.ndarray, persist: bool = False) -> "faiss.IndexFlatL2":  # type: ignore[name-defined]
    """Build a FAISS IndexFlatL2 from a 2D array of embeddings."""
    try:
        import faiss  # type: ignore[import-untyped]

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        if persist:
            faiss.write_index(index, str(FAISS_INDEX_PATH))
        return index
    except ImportError:
        logger.warning("faiss_not_installed_using_brute_force")
        return None  # type: ignore[return-value]


def search_similar(
    index: object,
    query: np.ndarray,
    k: int = 5,
    embeddings: np.ndarray | None = None,
) -> list[int]:
    """Return indices of k most similar rows."""
    try:
        import faiss  # type: ignore[import-untyped]

        if index is None:
            raise ImportError
        distances, indices = index.search(query.astype(np.float32).reshape(1, -1), k)
        return indices[0].tolist()
    except ImportError:
        if embeddings is None:
            return []
        diffs = np.linalg.norm(embeddings - query, axis=1)
        return np.argsort(diffs)[:k].tolist()
