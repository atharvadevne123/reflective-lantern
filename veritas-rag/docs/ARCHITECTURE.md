# Veritas-Rag Architecture — Scaling to 10 Million Documents

This document explains how each of the ten stages behaves at production scale,
and which in-process components swap for distributed infrastructure. The
Python implementation in `app/` is a faithful, fully-tested realization of the
architecture; every interface is designed so the scale-out backend replaces
the in-process one without touching the pipeline logic.

## Stage 1 — Ingestion & Normalization

| Concern | In-process | At 10M docs |
|---|---|---|
| Extraction | pypdf / stdlib email / html.parser | Worker fleet (queue-fed), same extractor code |
| Normalization | NFKC + whitespace + control chars | identical (pure function, embarrassingly parallel) |
| Exact dedup | SHA-256 dict | SHA-256 key in a KV store (Redis/Dynamo) |
| Near dedup | MinHash (64 perms) + LSH (16 bands) | identical algorithm; LSH buckets sharded by band hash |
| Versioning | in-memory chains | version table keyed by source identity |

Why it matters: at 10M documents, duplicate and near-duplicate content
(forwarded emails, re-exported PDFs, boilerplate) commonly makes up 20-40%
of a corpus. Dedup before indexing cuts index size, removes retrieval bias
toward repeated text, and prevents the same stale fact from dominating
results. LSH keeps near-dup detection O(1) per document instead of O(n).

## Stage 2 — Hybrid Retrieval (keyword + meaning)

BM25 catches exact terminology (error codes, section numbers, names) that
embedding models blur; vector search catches paraphrases BM25 cannot see.
Reciprocal-rank fusion combines the two ranked lists without score
calibration, and — critically — records *which retrievers* found each chunk,
a signal the trust scorer consumes later.

At scale, the `BM25Index` interface fronts OpenSearch/Tantivy shards and the
`VectorIndex` interface fronts a FAISS `IVF-PQ` (or HNSW) service. Both are
already id-addressed and support incremental add/remove for version churn.

## Stage 3 — ANN + Reranking (fast recall, then deep precision)

The retrieval funnel: 10M chunks → ~50 per retriever (ANN, milliseconds) →
~20 fused candidates → deep scorer → 5 evidence chunks. FAISS IVF is used
in-process the moment a corpus crosses the training threshold; the coarse
quantizer restricts each search to `nprobe` clusters, the standard trade
that makes 10M-vector search sub-10ms. The `Reranker` protocol ships a
deterministic lexical cross-scorer and is the slot for a neural
cross-encoder in production.

## Stage 4 — Trust Scoring

Every surviving chunk gets `trust = 0.30·freshness + 0.25·source_quality +
0.45·consistency`:

- **freshness** — exponential decay with configurable half-life; stale
  policy documents lose authority gradually rather than at a cliff.
- **source_quality** — priors by source type (a versioned PDF manual
  outranks a scraped page or an email thread).
- **consistency** — the strongest anti-hallucination signal: a chunk found
  independently by *both* retrievers, ranked high by both, and confirmed by
  the deep scorer is very unlikely to be a spurious match.

## Stages 5-7 — Constrained Generation, Citations, Fallback

The evidence gate runs *before* generation: if mean trust of the supporting
chunks is below threshold (or too few chunks clear the per-chunk floor), the
system answers "insufficient evidence" and never invokes the generator —
saving cost and making the refusal auditable.

Two generator backends satisfy one contract:

- **Extractive** (default, offline): answer sentences are copied verbatim
  from evidence — hallucination is structurally impossible.
- **Anthropic**: chunks are passed as `document` content blocks with native
  citations enabled; the system prompt forbids outside knowledge, treats
  document content as data (not instructions), and mandates the literal
  string `INSUFFICIENT EVIDENCE` when the excerpts don't answer. Claim-level
  citations come back as `char_location` spans mapped to chunk → doc → page.

A generator that produces no grounded sentences is itself converted into the
fallback answer — the "empty answer means no answer" invariant holds across
backends.

## Stage 8 — Continuous Evaluation

Three test modes run in CI (and are designed for a scheduled job in prod):

- **Recall/accuracy** on gold question→document pairs.
- **Hallucination**: questions the corpus cannot answer; *any* confident
  answer counts as a failure.
- **Adversarial**: prompt injection via query, instruction-smuggling
  documents, false presuppositions, fabricated-citation bait.

`scripts/run_eval.py` exits non-zero on any hallucination or adversarial
failure, so a regression blocks the merge.

## Stage 9 — Caching

Query traffic is power-law distributed; an LRU+TTL cache over normalized
queries serves the hot head without touching the indexes. Every index
mutation invalidates the cache wholesale — a stale answer citing a
superseded document version is treated as worse than a cache miss. At scale
the same keying scheme moves to Redis with per-tenant namespaces.

## Stage 10 — Observability

Every request produces a stage-by-stage trace: cache hit/miss, BM25 and
vector rankings, fusion candidates, rerank scores, per-chunk trust
breakdowns, the gate decision with its machine-readable reason, generation
attribution (answer → chunk ids), and per-stage latency. If the system
fails — wrong answer, missed evidence, unnecessary fallback — the trace
shows exactly which stage made the wrong call. In production the same trace
objects export to OpenTelemetry spans.

## Failure-mode map

| Symptom | Where the trace points |
|---|---|
| Wrong answer | generation.attribution → which chunk misled; trust components show why it was trusted |
| Missed known fact | bm25/vector stages → was the chunk retrieved at all? fusion → did it survive? |
| Unnecessary "insufficient evidence" | evidence_gate.reason + per-chunk trust breakdown |
| Slow query | stage_timings_ms — retrieval vs rerank vs generation |
| Stale answer | citations carry doc version; cache stats show TTL behavior |
