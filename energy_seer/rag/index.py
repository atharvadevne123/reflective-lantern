"""Build and persist FAISS index for energy pattern similarity search."""

from __future__ import annotations

import json
import logging

from rag.ingest import INDEX_PATH, META_PATH

logger = logging.getLogger(__name__)


def load_index():
    """Load the FAISS index from disk, returns (index, metadata) or (None, [])."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss not available")
        return None, []

    if not INDEX_PATH.exists():
        logger.warning("FAISS index not found at %s", INDEX_PATH)
        return None, []

    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else []
    logger.info("FAISS index loaded: %d vectors", index.ntotal)
    return index, meta


def rebuild_index(patterns: list[dict]) -> int:
    """Rebuild the FAISS index from scratch with the given patterns."""
    from rag.ingest import ingest_patterns

    return ingest_patterns(patterns)
