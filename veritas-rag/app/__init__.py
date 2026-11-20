"""Veritas-Rag: near-zero-hallucination RAG over 10M documents.

Pipeline stages:
    1.  Ingestion & normalization (dedup, PDF/email extraction, versioning)
    2.  Hybrid retrieval (BM25 keyword + semantic vector search)
    3.  ANN + reranking (fast candidate retrieval, then deep scoring)
    4.  Confidence scoring (per-chunk trust: freshness, source, consistency)
    5.  Constrained generation (answers only from retrieved evidence)
    6.  Citation-backed outputs (every claim traceable to doc + page)
    7.  Hallucination fallback (insufficient evidence below threshold)
    8.  Continuous evaluation (recall, adversarial, hallucination tests)
    9.  Caching (high-frequency retrieval cache)
    10. Observability (full trace: rankings, retrieval path, attribution)
"""

__version__ = "1.0.0"
