# Changelog

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
