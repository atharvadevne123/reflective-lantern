"""Threat intelligence corpus ingestion for RAG indexing."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path(__file__).parent / "corpus.json"


def load_corpus(path: Path | None = None) -> list[dict[str, str]]:
    """Load threat intel documents from JSON file.

    Args:
        path: Path to corpus JSON (list of {id, text} dicts). Defaults to
              the bundled corpus.json in this package.

    Returns:
        List of document dicts.
    """
    target = path or DEFAULT_CORPUS_PATH
    if not target.exists():
        logger.warning("Corpus file not found at %s; returning empty list", target)
        return []
    with target.open() as f:
        docs: list[dict[str, str]] = json.load(f)
    logger.info("Loaded %d documents from %s", len(docs), target)
    return docs


def validate_corpus(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Filter out malformed documents and return clean records."""
    valid = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if not doc.get("id") or not doc.get("text"):
            logger.debug("Skipping malformed doc: %s", doc)
            continue
        valid.append({"id": str(doc["id"]), "text": str(doc["text"])})
    logger.info("%d/%d documents passed validation", len(valid), len(docs))
    return valid
