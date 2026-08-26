"""Build and persist a FAISS-style vector index over threat intel documents."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INDEX_PATH = Path("rag_index.npz")


def build_tfidf_index(docs: list[dict[str, str]]) -> dict[str, Any]:
    """Compute TF-IDF matrix and vocabulary for a document collection.

    Args:
        docs: List of {id, text} dicts.

    Returns:
        Dict with 'matrix' (ndarray), 'vocab' (dict), and 'ids' (list).
    """
    texts = [d["text"].lower() for d in docs]
    all_words: set[str] = set()
    for t in texts:
        all_words.update(t.split())
    vocab = {w: i for i, w in enumerate(sorted(all_words))}

    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.split():
            if word in vocab:
                matrix[i, vocab[word]] += 1.0

    row_sums = matrix.sum(axis=1, keepdims=True) + 1e-9
    matrix /= row_sums

    # Smoothed IDF, as sklearn computes it: log((1 + n) / (1 + df)) + 1.
    # The naive log(n / (1 + df)) collapses to zero when a term appears in
    # exactly one document of a two-document corpus, and goes negative for
    # terms present in every document — both of which corrupt the ranking.
    df = (matrix > 0).sum(axis=0).astype(float)
    idf = np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0
    matrix *= idf

    logger.info("Index: %d docs × %d terms", len(texts), len(vocab))
    return {"matrix": matrix, "vocab": vocab, "ids": [d["id"] for d in docs]}


def save_index(index: dict[str, Any], path: Path = INDEX_PATH) -> None:
    """Persist the index to disk as a .npz archive."""
    np.savez(
        path,
        matrix=index["matrix"],
        ids=np.array(index["ids"]),
        vocab_keys=np.array(list(index["vocab"].keys())),
        vocab_vals=np.array(list(index["vocab"].values())),
    )
    logger.info("Index saved to %s", path)


def load_index(path: Path = INDEX_PATH) -> dict[str, Any] | None:
    """Load a persisted index from disk, or return None if absent."""
    if not path.exists():
        logger.warning("Index file not found at %s", path)
        return None
    data = np.load(path, allow_pickle=False)
    vocab = dict(zip(data["vocab_keys"].tolist(), data["vocab_vals"].tolist()))
    logger.info("Index loaded from %s", path)
    return {
        "matrix": data["matrix"],
        "vocab": vocab,
        "ids": data["ids"].tolist(),
    }
