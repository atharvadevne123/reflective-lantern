"""FAISS-based similar ticket retriever for RAG-augmented triage."""
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
INDEX_DIR = Path("rag_index")


def retrieve_similar(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return top-k most similar historical tickets for a given query string."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss not installed - returning empty results")
        return []

    index_path = INDEX_DIR / "tickets.index"
    meta_path = INDEX_DIR / "metadata.json"

    if not index_path.exists():
        logger.warning("FAISS index not found at %s", index_path)
        return []

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text())

    from rag.ingest import embed_text
    vec = embed_text([query]).astype(np.float32)
    faiss.normalize_L2(vec)

    distances, indices = index.search(vec, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0], strict=False):
        if idx < 0:
            continue
        entry = dict(metadata[idx])
        entry["similarity"] = round(float(dist), 4)
        results.append(entry)
    return results
