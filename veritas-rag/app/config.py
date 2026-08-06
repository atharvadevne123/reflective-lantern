"""Central configuration for Veritas-Rag, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class Settings:
    """Runtime settings.

    Attributes:
        embedding_dim: Dimensionality of the hashed embedding space.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between adjacent chunks in characters.
        bm25_top_k: Candidates pulled from the BM25 index per query.
        vector_top_k: Candidates pulled from the ANN index per query.
        rerank_top_k: Candidates that survive fusion and enter reranking.
        final_top_k: Chunks handed to the generator.
        confidence_threshold: Minimum mean trust score required to answer;
            below this the system returns "insufficient evidence".
        min_supporting_chunks: Minimum chunks above chunk_trust_floor needed.
        chunk_trust_floor: Per-chunk trust floor for a chunk to count as support.
        freshness_half_life_days: Trust halves for every this many days of age.
        cache_max_size: Maximum entries in the retrieval cache.
        cache_ttl_seconds: Retrieval cache time-to-live.
        trace_buffer_size: Number of recent request traces retained in memory.
        anthropic_model: Model used for constrained answer synthesis.
        generator_backend: "extractive" (offline, deterministic) or "anthropic".
        near_dup_threshold: Jaccard similarity above which docs are duplicates.
    """

    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 384))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    bm25_top_k: int = field(default_factory=lambda: _env_int("BM25_TOP_K", 50))
    vector_top_k: int = field(default_factory=lambda: _env_int("VECTOR_TOP_K", 50))
    rerank_top_k: int = field(default_factory=lambda: _env_int("RERANK_TOP_K", 20))
    final_top_k: int = field(default_factory=lambda: _env_int("FINAL_TOP_K", 5))

    confidence_threshold: float = field(
        default_factory=lambda: _env_float("CONFIDENCE_THRESHOLD", 0.45)
    )
    min_supporting_chunks: int = field(default_factory=lambda: _env_int("MIN_SUPPORTING_CHUNKS", 1))
    chunk_trust_floor: float = field(default_factory=lambda: _env_float("CHUNK_TRUST_FLOOR", 0.35))
    freshness_half_life_days: float = field(
        default_factory=lambda: _env_float("FRESHNESS_HALF_LIFE_DAYS", 365.0)
    )

    cache_max_size: int = field(default_factory=lambda: _env_int("CACHE_MAX_SIZE", 2048))
    cache_ttl_seconds: float = field(default_factory=lambda: _env_float("CACHE_TTL_SECONDS", 300.0))

    trace_buffer_size: int = field(default_factory=lambda: _env_int("TRACE_BUFFER_SIZE", 1000))

    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    )
    generator_backend: str = field(
        default_factory=lambda: os.getenv("GENERATOR_BACKEND", "extractive")
    )

    near_dup_threshold: float = field(
        default_factory=lambda: _env_float("NEAR_DUP_THRESHOLD", 0.85)
    )

    api_prefix: str = "/api/v1"
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
