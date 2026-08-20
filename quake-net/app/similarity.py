"""FAISS-backed similarity search over historical seismic event signatures.

Falls back to exact brute-force NumPy search when FAISS is unavailable, so the
service degrades gracefully rather than failing at import time.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "seismic_index.faiss"))

SIGNATURE_COLUMNS = [
    "depth_km",
    "p_wave_amplitude",
    "s_wave_amplitude",
    "epicentral_distance_km",
    "station_count",
]

try:  # pragma: no cover - exercised implicitly by whichever branch is installed
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False


def faiss_available() -> bool:
    """Report whether the native FAISS backend is importable."""
    return _FAISS_AVAILABLE


def build_signature_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    """Convert seismic event dicts into a float32 signature matrix.

    Args:
        records: Event dicts containing at least ``SIGNATURE_COLUMNS``.

    Returns:
        A ``(n_records, len(SIGNATURE_COLUMNS))`` float32 array.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("Cannot build a signature matrix from zero records")

    rows = [[float(rec.get(col, 0.0)) for col in SIGNATURE_COLUMNS] for rec in records]
    matrix = np.asarray(rows, dtype=np.float32)
    # Log-scale amplitude-like columns so distance is not dominated by outliers.
    return np.log1p(np.clip(matrix, 0.0, None))


class SeismicIndex:
    """Nearest-neighbour index over historical event signatures."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._payloads: list[dict[str, Any]] = []
        self._index: Any | None = None

    @property
    def dimension(self) -> int | None:
        """Vector width of the indexed signatures, or ``None`` when empty."""
        return None if self._matrix is None else int(self._matrix.shape[1])

    @property
    def size(self) -> int:
        """Number of indexed events."""
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def build(self, records: list[dict[str, Any]], *, persist: bool = False) -> SeismicIndex:
        """Index the given records, optionally writing the FAISS index to disk.

        Persistence is opt-in: an ad-hoc index built for a one-off query must
        never overwrite the index the API is serving from.
        """
        self._matrix = build_signature_matrix(records)
        self._payloads = list(records)

        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(self._matrix.shape[1])
            index.add(self._matrix)
            self._index = index
            if persist:
                faiss.write_index(index, str(INDEX_PATH))
                logger.info("Persisted FAISS index (%d vectors) to %s", self.size, INDEX_PATH)
        else:
            self._index = None
            logger.info("FAISS unavailable — using brute-force search over %d vectors", self.size)

        return self

    def search(self, query: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Return the ``k`` most similar historical events to ``query``.

        Raises:
            ValueError: If the index has not been built yet.
        """
        if self._matrix is None:
            raise ValueError("SeismicIndex.search called before build()")

        vector = build_signature_matrix([query])
        k = max(1, min(k, self.size))

        if self._index is not None:
            distances, indices = self._index.search(vector, k)
            pairs = zip(indices[0].tolist(), distances[0].tolist(), strict=True)
        else:
            deltas = self._matrix - vector
            squared = np.sum(deltas * deltas, axis=1)
            order = np.argsort(squared)[:k]
            pairs = zip(order.tolist(), squared[order].tolist(), strict=True)

        results: list[dict[str, Any]] = []
        for position, distance in pairs:
            if position < 0:
                continue
            payload = dict(self._payloads[position])
            payload["distance"] = round(float(distance), 4)
            payload["similarity"] = round(1.0 / (1.0 + float(distance)), 4)
            results.append(payload)
        return results


_index = SeismicIndex()


def get_index() -> SeismicIndex:
    """Return the process-wide seismic similarity index."""
    return _index
