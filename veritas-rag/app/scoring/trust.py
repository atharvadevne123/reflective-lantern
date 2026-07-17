"""Composite trust scoring for retrieved chunks.

Every chunk that survives reranking gets a trust score in [0, 1] built from:

    freshness            exponential decay on document age
    source_quality       prior by source type (configurable)
    retrieval_consistency agreement between the independent retrievers
                          plus the reranker's own relevance score

The answer-level confidence (mean trust of supporting chunks) is what the
hallucination fallback thresholds against.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from ..schemas import ScoredChunk

# Priors reflect how much a source type is trusted before content is examined.
SOURCE_QUALITY_PRIORS: dict[str, float] = {
    "pdf": 0.90,  # authored, versioned artifacts
    "markdown": 0.85,  # curated docs
    "text": 0.75,
    "html": 0.65,  # scraped content
    "email": 0.60,  # informal, frequently stale
}
_DEFAULT_SOURCE_QUALITY = 0.70

_WEIGHTS = {"freshness": 0.30, "source_quality": 0.25, "consistency": 0.45}


def freshness_score(
    created_at: str, half_life_days: float = 365.0, now: datetime | None = None
) -> float:
    """Exponential-decay freshness in (0, 1].

    Args:
        created_at: ISO 8601 timestamp of the source document.
        half_life_days: Age at which freshness has decayed to 0.5.
        now: Injection point for tests; defaults to current UTC time.
    """
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.5  # unknown age: neutral
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    age_days = max(0.0, (current - created).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / half_life_days)


def source_quality_score(source_type: str) -> float:
    """Source-type prior in [0, 1]."""
    return SOURCE_QUALITY_PRIORS.get(source_type, _DEFAULT_SOURCE_QUALITY)


def consistency_score(
    bm25_rank: int | None, vector_rank: int | None, rerank_score: float, top_k: int
) -> float:
    """Agreement between independent retrieval signals.

    A chunk found high in *both* the keyword and the semantic ranking, then
    confirmed by the deep scorer, is far less likely to be a spurious match
    than one surfaced by a single retriever.
    """

    def rank_credit(rank: int | None) -> float:
        if rank is None:
            return 0.0
        return max(0.0, 1.0 - (rank - 1) / top_k)

    both_found = 1.0 if (bm25_rank is not None and vector_rank is not None) else 0.0
    rank_signal = (rank_credit(bm25_rank) + rank_credit(vector_rank)) / 2.0
    return round(0.35 * both_found + 0.25 * rank_signal + 0.40 * min(1.0, rerank_score), 6)


def score_chunk(
    scored: ScoredChunk,
    half_life_days: float = 365.0,
    top_k: int = 50,
    now: datetime | None = None,
) -> ScoredChunk:
    """Attach trust and its component breakdown to a scored chunk."""
    fresh = freshness_score(scored.chunk.created_at, half_life_days, now=now)
    quality = source_quality_score(scored.chunk.source_type)
    consistency = consistency_score(
        scored.bm25_rank, scored.vector_rank, scored.rerank_score, top_k
    )

    scored.trust_components = {
        "freshness": round(fresh, 6),
        "source_quality": round(quality, 6),
        "consistency": consistency,
    }
    scored.trust = round(
        _WEIGHTS["freshness"] * fresh
        + _WEIGHTS["source_quality"] * quality
        + _WEIGHTS["consistency"] * consistency,
        6,
    )
    return scored


def answer_confidence(chunks: list[ScoredChunk]) -> float:
    """Mean trust of the supporting chunks (0.0 for an empty set)."""
    if not chunks:
        return 0.0
    return round(sum(c.trust for c in chunks) / len(chunks), 6)
