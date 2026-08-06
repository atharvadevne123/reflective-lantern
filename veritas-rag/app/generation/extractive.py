"""Offline extractive generator — zero hallucination by construction.

Every answer sentence is copied verbatim from a retrieved chunk, so the
answer cannot contain content absent from the evidence. This is the default
backend for dev, CI, and air-gapped deployments; the Anthropic backend
(llm.py) produces fluent abstractive answers under the same grounding
contract.
"""

from __future__ import annotations

import re

from ..retrieval.bm25 import tokenize
from ..schemas import Citation, ScoredChunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_SUFFIXES = ("ment", "ing", "ed", "es", "ly", "s")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _stem(token: str) -> str:
    """Light suffix stripper so 'reimbursed' matches 'reimbursement'."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    return token.rstrip("e")


def _stemmed(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text)}


class ExtractiveGenerator:
    """Select the most query-relevant sentences from the evidence."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: str, chunks: list[ScoredChunk]) -> tuple[str, list[Citation]]:
        """Build an answer from verbatim evidence sentences.

        Args:
            query: The user question.
            chunks: Trust-scored supporting chunks, best first.

        Returns:
            (answer_text, citations); empty answer when nothing overlaps
            the query enough to be quoted.
        """
        query_terms = _stemmed(query)
        if not query_terms:
            return "", []

        # A single shared term is not evidence of relevance: require two
        # stemmed content-term matches unless the query itself is tiny.
        min_overlap = 1 if len(query_terms) <= 2 else 2

        candidates: list[tuple[float, str, ScoredChunk]] = []
        for scored in chunks:
            for sentence in _sentences(scored.chunk.text):
                terms = _stemmed(sentence)
                overlap = len(terms & query_terms)
                if overlap < min_overlap:
                    continue
                # Rank sentences by query overlap weighted by chunk trust
                relevance = overlap / max(1, len(query_terms)) * (0.5 + scored.trust)
                candidates.append((relevance, sentence, scored))

        if not candidates:
            return "", []

        candidates.sort(key=lambda item: item[0], reverse=True)

        picked: list[tuple[str, ScoredChunk]] = []
        seen: set[str] = set()
        for _, sentence, scored in candidates:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((sentence, scored))
            if len(picked) >= self.max_sentences:
                break

        answer = " ".join(sentence for sentence, _ in picked)
        citations = [
            Citation(
                claim=sentence,
                chunk_id=scored.chunk.chunk_id,
                doc_id=scored.chunk.doc_id,
                source=scored.chunk.source,
                page=scored.chunk.page,
                quote=sentence,
            )
            for sentence, scored in picked
        ]
        return answer, citations
