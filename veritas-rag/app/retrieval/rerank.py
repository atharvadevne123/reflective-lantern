"""Stage-2 deep scoring of fused candidates.

The fast retrievers (BM25 + ANN) optimize recall over millions of chunks;
the reranker spends more compute on the few dozen survivors to optimize
precision. The default implementation is a lexical cross-scorer combining
term coverage, ordered-window proximity, and length calibration — pluggable
for a neural cross-encoder in production via the `Reranker` protocol.
"""

from __future__ import annotations

from typing import Protocol

from .bm25 import tokenize


class Reranker(Protocol):
    """Interface for deep scorers."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score in [0, 1] per candidate text."""
        ...


class LexicalReranker:
    """Deterministic query-document cross-scorer."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each candidate against the query.

        Components:
            coverage: fraction of unique query terms present in the text.
            proximity: 1 / span of the tightest window containing the most
                query terms (rewards terms appearing close together).
            density: query-term hits per text token (length calibration).
        """
        query_terms = tokenize(query)
        unique_terms = set(query_terms)
        if not unique_terms:
            return [0.0] * len(texts)

        scores: list[float] = []
        for text in texts:
            tokens = tokenize(text)
            if not tokens:
                scores.append(0.0)
                continue

            positions = [i for i, tok in enumerate(tokens) if tok in unique_terms]
            present = {tokens[i] for i in positions}
            coverage = len(present) / len(unique_terms)

            proximity = 0.0
            if len(positions) >= 2:
                # Tightest window holding min(len(present), 3) query hits
                need = min(len(present), 3)
                best_span = None
                for i in range(len(positions) - need + 1):
                    span = positions[i + need - 1] - positions[i] + 1
                    if best_span is None or span < best_span:
                        best_span = span
                if best_span:
                    proximity = need / best_span
            elif len(positions) == 1:
                proximity = 0.5

            density = min(1.0, len(positions) / max(8, len(tokens)) * 4)

            scores.append(round(0.6 * coverage + 0.25 * min(1.0, proximity) + 0.15 * density, 6))
        return scores
