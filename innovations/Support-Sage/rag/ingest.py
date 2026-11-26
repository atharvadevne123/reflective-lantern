"""Ingest historical tickets into the FAISS vector index."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INDEX_DIR = Path("rag_index")


def embed_text(texts: list[str]) -> np.ndarray:
    """Lightweight TF-IDF + SVD embedding for ticket text (no external API needed)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    embedder_path = INDEX_DIR / "embedder.joblib"
    import joblib

    if embedder_path.exists():
        embedder = joblib.load(embedder_path)
        return embedder.transform(texts).astype(np.float32)

    embedder = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=128, random_state=42)),
        ]
    )
    vectors = embedder.fit_transform(texts).astype(np.float32)
    INDEX_DIR.mkdir(exist_ok=True)
    joblib.dump(embedder, embedder_path)
    return vectors


def ingest_tickets(df: pd.DataFrame) -> int:
    """Build FAISS index from a DataFrame with subject/description/priority/category columns."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss not installed - RAG indexing skipped")
        return 0

    INDEX_DIR.mkdir(exist_ok=True)
    texts = [(str(r.get("subject", "")) + " " + str(r.get("description", ""))).strip() for _, r in df.iterrows()]
    vectors = embed_text(texts)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_DIR / "tickets.index"))

    metadata = df[["subject", "priority", "category"]].fillna("").to_dict(orient="records")
    import json

    (INDEX_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("FAISS index built with %d tickets (dim=%d)", len(df), dim)
    return len(df)
