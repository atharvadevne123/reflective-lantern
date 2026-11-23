"""Stage 1 — ingestion and normalization."""

from .chunker import chunk_document
from .dedup import DuplicateDetector
from .extractors import detect_source_type, extract_text
from .normalize import normalize_text
from .versioning import VersionStore

__all__ = [
    "DuplicateDetector",
    "VersionStore",
    "chunk_document",
    "detect_source_type",
    "extract_text",
    "normalize_text",
]
