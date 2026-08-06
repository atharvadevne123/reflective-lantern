"""FAISS nearest-neighbour anomaly detection for manufacturing sensor data.

Uses cosine similarity against a reference index of healthy-production vectors
to flag anomalous readings before they reach the ensemble model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = Path("faiss_index.bin")
REF_VECTORS_PATH = Path("faiss_ref_vectors.npy")

_index = None
_ref_vectors: np.ndarray | None = None


def build_index(vectors: np.ndarray, save: bool = True) -> None:
    """Build and optionally persist a FAISS flat-L2 index from reference vectors."""
    global _index, _ref_vectors
    try:
        import faiss  # type: ignore[import]

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors.astype(np.float32))
        _index = index
        _ref_vectors = vectors

        if save:
            faiss.write_index(index, str(INDEX_PATH))
            np.save(str(REF_VECTORS_PATH), vectors)
            logger.info("FAISS index saved: %d vectors, dim=%d", len(vectors), dim)
    except ImportError:
        logger.warning("faiss-cpu not installed — anomaly detection disabled.")


def load_index() -> bool:
    """Load a persisted FAISS index from disk. Returns True on success."""
    global _index, _ref_vectors
    try:
        import faiss  # type: ignore[import]

        if INDEX_PATH.exists() and REF_VECTORS_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
            _ref_vectors = np.load(str(REF_VECTORS_PATH))
            logger.info("FAISS index loaded: %d vectors", _index.ntotal)
            return True
    except ImportError:
        logger.warning("faiss-cpu not installed — anomaly detection disabled.")
    except Exception:
        logger.exception("Failed to load FAISS index.")
    return False


def is_anomalous(query: np.ndarray, k: int = 5, threshold: float = 50.0) -> dict:
    """Return True if the nearest-neighbour distance exceeds the threshold.

    A high L2 distance indicates the query point is far from all healthy
    reference vectors and may represent an anomalous operating condition.
    """
    if _index is None:
        return {"anomalous": False, "nn_distance": None, "nn_k": k}

    q = query.astype(np.float32).reshape(1, -1)
    distances, _ = _index.search(q, k)
    avg_dist = float(distances[0].mean())
    return {
        "anomalous": avg_dist > threshold,
        "nn_distance": round(avg_dist, 4),
        "nn_k": k,
    }
