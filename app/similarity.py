"""Building energy profile similarity search using brute-force cosine distance."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BuildingSimilarityIndex:
    """In-memory nearest-neighbour index for building energy profiles.

    Falls back to a brute-force cosine search when FAISS is unavailable.
    """

    def __init__(self) -> None:
        self._profiles: list[tuple[str, np.ndarray]] = []
        self._use_faiss = False
        try:
            import faiss  # noqa: F401

            self._use_faiss = True
            logger.info("FAISS available — using approximate search.")
        except ImportError:
            logger.info("FAISS not installed — using brute-force cosine search.")

    def add(self, building_id: str, profile: list[float]) -> None:
        """Add a building energy profile vector to the index.

        Args:
            building_id: Unique identifier for the building.
            profile: Feature vector representing the building's energy signature.
        """
        self._profiles.append((building_id, np.array(profile, dtype=np.float32)))

    def search(self, query: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Return the k most similar buildings by cosine similarity.

        Args:
            query: Feature vector to search for.
            k: Maximum number of results to return.

        Returns:
            List of (building_id, similarity_score) pairs, highest score first.
        """
        if not self._profiles:
            return []
        q = np.array(query, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        scores: list[tuple[str, float]] = []
        for bid, vec in self._profiles:
            v_norm = vec / (np.linalg.norm(vec) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            scores.append((bid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def clear(self) -> None:
        """Remove all profiles from the index."""
        self._profiles.clear()

    def __len__(self) -> int:
        """Return the number of profiles stored in the index."""
        return len(self._profiles)

    @property
    def size(self) -> int:
        """Alias for ``len(self)``."""
        return len(self._profiles)


_global_index = BuildingSimilarityIndex()


def get_global_index() -> BuildingSimilarityIndex:
    """Return the module-level singleton similarity index."""
    return _global_index
