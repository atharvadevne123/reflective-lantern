"""Central configuration for Veritas-Rag, loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    """Read an environment variable as a float, returning *default* if unset."""
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    """Read an environment variable as an int, returning *default* if unset."""
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

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {self.embedding_dim}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
        if not 0.0 <= self.chunk_trust_floor < 1.0:
            raise ValueError(
                f"chunk_trust_floor must be in [0, 1), got {self.chunk_trust_floor}"
            )
        if self.freshness_half_life_days <= 0.0:
            raise ValueError(
                f"freshness_half_life_days must be > 0, got {self.freshness_half_life_days}"
            )
        if not 0.0 < self.near_dup_threshold <= 1.0:
            raise ValueError(
                f"near_dup_threshold must be in (0, 1], got {self.near_dup_threshold}"
            )
        if self.final_top_k < 1:
            raise ValueError(f"final_top_k must be >= 1, got {self.final_top_k}")


settings = Settings()


def settings_summary(s: Settings) -> dict[str, object]:
    """Return a human-readable summary of key settings for logging/introspection.

    Args:
        s: Settings instance to summarize.

    Returns:
        Dict of the most operationally-relevant configuration values.
    """
    return {
        "embedding_dim": s.embedding_dim,
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "bm25_top_k": s.bm25_top_k,
        "vector_top_k": s.vector_top_k,
        "final_top_k": s.final_top_k,
        "confidence_threshold": s.confidence_threshold,
        "chunk_trust_floor": s.chunk_trust_floor,
        "generator_backend": s.generator_backend,
        "near_dup_threshold": s.near_dup_threshold,
    }
