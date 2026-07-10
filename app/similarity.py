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
        """Add a building energy profile vector to the index."""
        self._profiles.append((building_id, np.array(profile, dtype=np.float32)))

    def search(self, query: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Return the k most similar buildings by cosine similarity."""
        if not self._profiles:
            return []
        q = np.array(query, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        scores = []
        for bid, vec in self._profiles:
            v_norm = vec / (np.linalg.norm(vec) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            scores.append((bid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def __len__(self) -> int:
        return len(self._profiles)


_global_index = BuildingSimilarityIndex()
