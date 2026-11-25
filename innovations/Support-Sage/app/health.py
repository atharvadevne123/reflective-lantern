"""Detailed health check utilities."""
from __future__ import annotations

from pathlib import Path


def check_model_files() -> dict[str, bool]:
    """Return whether required model files exist on disk."""
    return {
        "priority_model": Path("models/priority_model.joblib").exists(),
        "priority_encoder": Path("models/priority_encoder.joblib").exists(),
        "resolution_model": Path("models/resolution_model.joblib").exists(),
        "metrics": Path("models/metrics.json").exists(),
    }

def check_rag_index() -> dict[str, bool]:
    """Return whether FAISS index files exist."""
    return {
        "faiss_index": Path("rag_index/tickets.index").exists(),
        "metadata": Path("rag_index/metadata.json").exists(),
    }
