"""FAISS-backed runbook vector index for RAG-style retrieval in Ops-Vision."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Runbook:
    """A single SRE runbook entry.

    Attributes:
        title: Short runbook title.
        content: Full runbook text.
        category: Category tag (e.g., "cpu", "memory", "network").
    """

    title: str
    content: str
    category: str


class RunbookIndex:
    """TF-IDF + FAISS runbook retrieval index.

    Builds a lightweight term-frequency embedding for each runbook and
    stores them in a FAISS flat index for approximate nearest-neighbour
    search.  Falls back to cosine-similarity brute-force when faiss-cpu
    is unavailable.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialise an empty index.

        Args:
            embedding_dim: Dimensionality of the TF-IDF projection vector.
        """
        self.embedding_dim = embedding_dim
        self.runbooks: list[Runbook] = []
        self._index: object | None = None
        self._embeddings: np.ndarray | None = None
        self._vocab: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase and split text into tokens, stripping punctuation."""
        import re

        return re.findall(r"[a-z]+", text.lower())

    def _build_vocab(self, corpus: list[str]) -> None:
        """Build vocabulary from corpus, keeping the top embedding_dim terms."""
        from collections import Counter

        counts: Counter = Counter()
        for doc in corpus:
            counts.update(self._tokenize(doc))
        top_terms = [term for term, _ in counts.most_common(self.embedding_dim)]
        self._vocab = {term: i for i, term in enumerate(top_terms)}

    def _embed(self, text: str) -> np.ndarray:
        """Produce a normalised TF vector for text using the learned vocab.

        Args:
            text: Input text to embed.

        Returns:
            L2-normalised float32 vector of shape (embedding_dim,).
        """
        tokens = self._tokenize(text)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for token in tokens:
            if token in self._vocab:
                vec[self._vocab[token]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def build(self, runbooks: list[Runbook]) -> None:
        """Build the FAISS index from a list of runbooks.

        Args:
            runbooks: Runbook objects to index.
        """
        self.runbooks = runbooks
        corpus = [f"{r.title} {r.content}" for r in runbooks]
        self._build_vocab(corpus)
        embeddings = np.stack([self._embed(c) for c in corpus])
        self._embeddings = embeddings

        try:
            import faiss

            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(embeddings)
            self._index = index
            logger.info("FAISS index built with %d runbooks (faiss-cpu)", len(runbooks))
        except ImportError:
            logger.warning("faiss-cpu not installed — using brute-force cosine search")
            self._index = None

    def search(self, query: str, top_k: int = 3) -> list[tuple[Runbook, float]]:
        """Search the index for runbooks similar to query.

        Args:
            query: Natural-language search string.
            top_k: Number of results to return.

        Returns:
            List of (Runbook, score) tuples sorted by descending similarity.
        """
        if not self.runbooks:
            return []

        q_vec = self._embed(query).reshape(1, -1)

        if self._index is not None:
            distances, indices = self._index.search(q_vec, min(top_k, len(self.runbooks)))
            results = [
                (self.runbooks[idx], float(dist))
                for idx, dist in zip(indices[0], distances[0], strict=False)
                if idx >= 0
            ]
        else:
            assert self._embeddings is not None
            scores = (self._embeddings @ q_vec.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [(self.runbooks[int(i)], float(scores[i])) for i in top_indices]

        return results

    def save(self, path: Path) -> None:
        """Persist the index metadata to a JSON file (embeddings not saved).

        Args:
            path: Destination JSON file path.
        """
        data = {
            "runbooks": [
                {"title": r.title, "content": r.content, "category": r.category}
                for r in self.runbooks
            ],
            "vocab": self._vocab,
            "embedding_dim": self.embedding_dim,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("RunbookIndex metadata saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "RunbookIndex":
        """Load a RunbookIndex from a JSON metadata file.

        Args:
            path: JSON file previously written by save().

        Returns:
            Populated RunbookIndex ready for search.
        """
        data = json.loads(path.read_text())
        index = cls(embedding_dim=data["embedding_dim"])
        index._vocab = data["vocab"]
        runbooks = [Runbook(r["title"], r["content"], r["category"]) for r in data["runbooks"]]
        index.build(runbooks)
        logger.info("RunbookIndex loaded from %s (%d runbooks)", path, len(runbooks))
        return index


_index_singleton: RunbookIndex | None = None


def get_runbook_index(runbooks_path: str = "data/runbooks/sample_runbooks.json") -> RunbookIndex:
    """Return (or lazily build) the global RunbookIndex singleton.

    Args:
        runbooks_path: Path to JSON file containing runbook definitions.

    Returns:
        Populated RunbookIndex.
    """
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = RunbookIndex()
        try:
            data = json.loads(Path(runbooks_path).read_text())
            runbooks = [Runbook(r["title"], r["content"], r["category"]) for r in data]
            _index_singleton.build(runbooks)
        except Exception:
            logger.exception("Failed to load runbooks from %s — index empty", runbooks_path)
    return _index_singleton
