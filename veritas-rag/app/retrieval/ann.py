"""Approximate-nearest-neighbour vector index (FAISS, numpy fallback).

Small corpora use an exact inner-product search. Once the corpus crosses
`ivf_threshold`, the index is rebuilt as FAISS IVF-Flat — the coarse quantizer
prunes the search to a fraction of the corpus, which is what makes 10M-vector
retrieval answerable in milliseconds.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    _FAISS = True
except ImportError:  # pragma: no cover - exercised only without faiss
    _FAISS = False


class VectorIndex:
    """ANN index over chunk embeddings with id mapping and deletion support."""

    def __init__(self, dim: int, ivf_threshold: int = 5000) -> None:
        """Initialise the index for *dim*-dimensional embeddings, using FAISS IVF above *ivf_threshold* documents."""
        self.dim = dim
        self.ivf_threshold = ivf_threshold
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._deleted: set[str] = set()
        self._faiss_index = None
        self._dirty = True

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        """Append vectors for the given chunk ids."""
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError("chunk_ids and vectors length mismatch")
        self._ids.extend(chunk_ids)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._dirty = True

    def remove(self, chunk_ids: list[str]) -> None:
        """Mark chunk ids as deleted (filtered from results, compacted on rebuild)."""
        self._deleted.update(chunk_ids)

    def __len__(self) -> int:
        """Return the count of active (non-deleted) chunks in the index."""
        return len(self._ids) - len(self._deleted & set(self._ids))

    def _rebuild_faiss(self) -> None:
        """Rebuild the FAISS IVF index from current non-deleted vectors."""
        n = self._vectors.shape[0]
        if not _FAISS or n < self.ivf_threshold:
            self._faiss_index = None
            self._dirty = False
            return

        nlist = max(8, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(self._vectors)
        index.add(self._vectors)
        index.nprobe = max(1, nlist // 8)
        self._faiss_index = index
        self._dirty = False
        logger.info("Rebuilt FAISS IVF index: n=%d nlist=%d", n, nlist)

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, cosine_score) nearest to the query vector."""
        if self._vectors.shape[0] == 0:
            return []

        if self._dirty:
            self._rebuild_faiss()

        query = query_vector.astype(np.float32).reshape(1, -1)
        # Over-fetch to compensate for deleted entries filtered post-hoc
        fetch = min(self._vectors.shape[0], top_k + len(self._deleted))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, fetch)
            pairs = [
                (int(i), float(s)) for i, s in zip(indices[0], scores[0], strict=True) if i >= 0
            ]
        else:
            sims = (self._vectors @ query.T).ravel()
            order = np.argsort(-sims)[:fetch]
            pairs = [(int(i), float(sims[i])) for i in order]

        results: list[tuple[str, float]] = []
        for idx, score in pairs:
            chunk_id = self._ids[idx]
            if chunk_id in self._deleted:
                continue
            results.append((chunk_id, score))
            if len(results) >= top_k:
                break
        return results
