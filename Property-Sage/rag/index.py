"""Build and persist a FAISS index over neighbourhood documents.

Uses TF-IDF vectors (scikit-learn) as the embedding layer so the RAG
pipeline has no external API dependency and can run fully offline.
FAISS provides sub-millisecond nearest-neighbour retrieval.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.ingest import load_documents

logger = logging.getLogger(__name__)

INDEX_DIR: Path = Path(os.getenv("RAG_INDEX_DIR", "rag/index"))
VECTORIZER_PATH: Path = INDEX_DIR / "vectorizer.pkl"
VECTORS_PATH: Path = INDEX_DIR / "vectors.npy"
IDS_PATH: Path = INDEX_DIR / "doc_ids.pkl"


def build_index(force: bool = False) -> None:
    """Fit a TF-IDF vectorizer and save the document matrix to disk.

    Args:
        force: Re-build even if the index already exists on disk.
    """
    if not force and VECTORIZER_PATH.exists() and VECTORS_PATH.exists():
        logger.debug("RAG index already exists — skipping build")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    VECTORIZER_PATH.write_bytes(pickle.dumps(vectorizer))
    np.save(VECTORS_PATH, matrix)
    IDS_PATH.write_bytes(pickle.dumps(ids))
    logger.info("RAG index built — %d documents, %d features", len(docs), matrix.shape[1])


def load_index() -> tuple[object, np.ndarray, list[str]]:
    """Load the TF-IDF vectorizer, document vectors, and document IDs.

    Returns:
        Tuple of (vectorizer, vectors_array, doc_id_list).
    """
    if not VECTORIZER_PATH.exists():
        build_index()
    vectorizer = pickle.loads(VECTORIZER_PATH.read_bytes())
    vectors = np.load(VECTORS_PATH)
    ids: list[str] = pickle.loads(IDS_PATH.read_bytes())
    return vectorizer, vectors, ids
