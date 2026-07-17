"""Reciprocal-rank fusion of BM25 (keyword) and vector (meaning) results."""

from __future__ import annotations


def reciprocal_rank_fusion(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float, int | None, int | None]]:
    """Fuse two ranked lists with RRF.

    RRF is rank-based, so the incommensurable BM25 and cosine score scales
    never need calibration. Chunks found by *both* retrievers accumulate
    two reciprocal terms and rise to the top — exactly the agreement signal
    the trust scorer later reuses.

    Args:
        bm25_results: (chunk_id, score) from the keyword index.
        vector_results: (chunk_id, score) from the ANN index.
        k: RRF damping constant (60 is the literature default).

    Returns:
        List of (chunk_id, fused_score, bm25_rank, vector_rank), best first.
        Ranks are 1-indexed, None when a retriever did not return the chunk.
    """
    bm25_rank = {cid: rank for rank, (cid, _) in enumerate(bm25_results, start=1)}
    vector_rank = {cid: rank for rank, (cid, _) in enumerate(vector_results, start=1)}

    fused: dict[str, float] = {}
    for cid, rank in bm25_rank.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    for cid, rank in vector_rank.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(cid, score, bm25_rank.get(cid), vector_rank.get(cid)) for cid, score in ranked]
