# Changelog

## [1.1.0] — 2026-10-22

### Added
- `query_batch()` on `RagPipeline` for processing multiple questions in one call.
- `sources()` on `RagPipeline` to list all indexed document source names.
- `GET /api/v1/sources` endpoint returning all unique source names.
- `settings_summary()` in `config.py` for runtime configuration introspection.
- `cache_age_seconds()` helper for cache entry age inspection.
- `normalize_trust_scores()` in `scoring/trust.py` for relative score rescaling.
- `truncate_to_bytes()` in `ingestion/normalize.py` for safe byte-bounded text.
- `trust_tier()`, `weighted_answer_confidence()`, `min_trust_gate()`, `trust_variance()`,
  `above_threshold_ratio()`, `top_k_trust()` scoring utilities in `scoring/trust.py`.
- `__repr__` on `ScoredChunk` for clearer debug output.
- `/health` now exposes a `config` summary field.
- Two new adversarial test cases: `jailbreak_roleplay` and `empty_query_probe`.
- Structured logging throughout: `fallback.py`, `extractive.py`, `dedup.py`, `chunker.py`,
  `versioning.py`, `rerank.py`, `hybrid.py`.
- 50+ new tests covering new utilities, edge cases, and endpoints.

### Changed
- Improved docstrings on `BM25Index`, `VectorIndex`, `HashingEmbedder`, `LexicalReranker`.
- Added `__all__` exports to `schemas.py` and `retrieval/hybrid.py`.
- `pyproject.toml`: expanded classifiers, added `dev` optional-dependency group.

## [1.0.0] — 2026-07-17

### Added
- Stage 1: PDF/email/HTML/text extraction, NFKC normalization, SHA-256 exact
  dedup, MinHash+LSH near-dedup, per-source version chains with index retirement.
- Stage 2: Okapi BM25 keyword index and hashed-feature semantic embedder (pluggable).
- Stage 3: FAISS IVF ANN index with numpy fallback; lexical cross-scorer reranker (pluggable).
- Stage 4: per-chunk trust scoring (freshness decay, source-type priors, retrieval consistency).
- Stage 5: constrained generation contract; extractive backend (verbatim evidence) and
  Anthropic backend with native citations.
- Stage 6: claim-level citations carrying chunk, document, source, and page.
- Stage 7: pre-generation evidence gate + insufficient-evidence fallback.
- Stage 8: evaluation harness (recall@k, hallucination rate, adversarial suite) with CI gate.
- Stage 9: LRU+TTL retrieval cache with normalized keys and mutation invalidation.
- Stage 10: per-request stage traces with rankings, trust breakdowns, attribution, latencies.
- FastAPI service, Docker/compose, 108-test suite, architecture docs and diagram.
