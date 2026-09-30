"""FAISS-based similar ticket retrieval for RAG-style context augmentation."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore[import]
    _FAISS_AVAILABLE = True
except ImportError:
    pass


class TicketRetriever:
    """In-memory FAISS index for retrieving similar historical tickets.

    Falls back to cosine similarity via numpy when faiss is not installed.
    """

    def __init__(self, dim: int = 218) -> None:
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._index = None
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
            logger.debug("FAISS IndexFlatIP initialized (dim=%d)", dim)

    def add(self, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        """Add a ticket embedding and its metadata to the index."""
        vec = vector.reshape(1, -1).astype(np.float32)
        if _FAISS_AVAILABLE and self._index is not None:
            faiss.normalize_L2(vec)
            self._index.add(vec)
        else:
            self._vectors.append(vec.flatten())
        self._metadata.append(metadata)

    def search(self, query: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        """Return the k most similar tickets to the query vector.

        Args:
            query: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of metadata dicts ordered by similarity, highest first.
        """
        if not self._metadata:
            return []

        k = min(k, len(self._metadata))
        qvec = query.reshape(1, -1).astype(np.float32)

        if _FAISS_AVAILABLE and self._index is not None:
            faiss.normalize_L2(qvec)
            _, idxs = self._index.search(qvec, k)
            return [self._metadata[i] for i in idxs[0] if i >= 0]

        # numpy fallback
        matrix = np.stack(self._vectors, axis=0)
        q_norm = qvec.flatten() / (np.linalg.norm(qvec) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        sims = (matrix / norms) @ q_norm
        top_idxs = np.argsort(sims)[::-1][:k]
        return [self._metadata[i] for i in top_idxs]

    @property
    def size(self) -> int:
        return len(self._metadata)
