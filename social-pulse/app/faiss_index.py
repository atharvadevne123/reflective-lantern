"""FAISS-based similarity search for finding posts similar to a query."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import]
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    logger.info("faiss not installed; similarity search disabled")


@dataclass
class SimilarPost:
    """A single result from a similarity search."""

    index: int
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class PostIndex:
    """Flat L2 FAISS index for post embedding vectors.

    Falls back to brute-force cosine search when faiss is not installed.
    """

    def __init__(self, dim: int = 217) -> None:
        """Initialise an empty index with the given feature dimensionality."""
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._index: Any = None
        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(dim)
            logger.info("FAISS IndexFlatL2 initialised (dim=%d)", dim)
        else:
            logger.info("Using brute-force fallback (faiss not available)")

    def add(self, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Add a single feature vector to the index.

        Args:
            vector: 1-D float32 array of length *dim*.
            metadata: Optional dict stored alongside the vector.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vec.shape[1]}")
        self._vectors.append(vec[0])
        self._metadata.append(metadata or {})
        if _HAS_FAISS and self._index is not None:
            self._index.add(vec)

    def search(self, query: np.ndarray, k: int = 5) -> list[SimilarPost]:
        """Return the *k* most similar posts to *query*.

        Args:
            query: 1-D float32 feature vector.
            k: Number of nearest neighbours to return.

        Returns:
            List of SimilarPost sorted by ascending distance.
        """
        if not self._vectors:
            return []

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        k_actual = min(k, len(self._vectors))

        if _HAS_FAISS and self._index is not None:
            distances, indices = self._index.search(q, k_actual)
            return [
                SimilarPost(index=int(idx), distance=float(dist), metadata=self._metadata[idx])
                for dist, idx in zip(distances[0], indices[0])
                if idx >= 0
            ]

        # Brute-force L2 fallback
        matrix = np.stack(self._vectors)
        diffs = matrix - q
        dists = np.sum(diffs ** 2, axis=1)
        nearest = np.argsort(dists)[:k_actual]
        return [
            SimilarPost(index=int(i), distance=float(dists[i]), metadata=self._metadata[i])
            for i in nearest
        ]

    def save(self, path: Path) -> None:
        """Persist the index and metadata to *path*."""
        data = {"vectors": self._vectors, "metadata": self._metadata, "dim": self.dim}
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        if _HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, str(path) + ".faiss")
        logger.info("Index saved to %s (%d vectors)", path, len(self._vectors))

    @classmethod
    def load(cls, path: Path) -> "PostIndex":
        """Load a previously saved index from *path*."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        inst = cls(dim=data["dim"])
        inst._vectors = data["vectors"]
        inst._metadata = data["metadata"]
        if _HAS_FAISS:
            faiss_path = str(path) + ".faiss"
            if Path(faiss_path).exists():
                inst._index = faiss.read_index(faiss_path)
            else:
                inst._index = faiss.IndexFlatL2(inst.dim)
                if inst._vectors:
                    matrix = np.stack(inst._vectors).astype(np.float32)
                    inst._index.add(matrix)
        logger.info("Index loaded from %s (%d vectors)", path, len(inst._vectors))
        return inst

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return len(self._vectors)
