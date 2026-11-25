"""RAG-powered similar ticket retrieval endpoint."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["rag"])

@router.post("/similar", summary="Find similar historical tickets")
async def find_similar(payload: dict[str, str]) -> dict[str, Any]:
    """Return top-5 historically similar tickets using FAISS retrieval."""
    from rag.retriever import retrieve_similar
    query = payload.get("subject", "") + " " + payload.get("description", "")
    results = retrieve_similar(query.strip(), top_k=5)
    return {"query": query.strip(), "similar_tickets": results, "count": len(results)}
