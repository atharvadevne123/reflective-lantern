"""Okapi BM25 keyword index (exact term matching)."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Classic Okapi BM25 over chunk texts.

    Incremental adds; deletions by id set. At 10M-document scale this module
    is the interface contract for an external engine (OpenSearch / Tantivy);
    the in-process implementation serves dev and evaluation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, Counter[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._total_len = 0

    def add(self, chunk_id: str, text: str) -> None:
        """Index a chunk."""
        tokens = Counter(tokenize(text))
        self._doc_tokens[chunk_id] = tokens
        length = sum(tokens.values())
        self._doc_len[chunk_id] = length
        self._total_len += length
        for term in tokens:
            self._df[term] += 1

    def remove(self, chunk_id: str) -> None:
        """Drop a chunk from the index (used when versions supersede)."""
        tokens = self._doc_tokens.pop(chunk_id, None)
        if tokens is None:
            return
        self._total_len -= self._doc_len.pop(chunk_id, 0)
        for term in tokens:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) for the query.

        Args:
            query: Natural-language query.
            top_k: Maximum results.
        """
        if not self._doc_tokens:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        n_docs = len(self._doc_tokens)
        avg_len = self._total_len / n_docs
        scores: dict[str, float] = {}

        for term in set(query_terms):
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for chunk_id, tokens in self._doc_tokens.items():
                tf = tokens.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self._doc_len[chunk_id] / avg_len)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]
