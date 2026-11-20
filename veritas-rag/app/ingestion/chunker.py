"""Page-aware chunking with overlap."""

from __future__ import annotations

import re

from ..schemas import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split a document into overlapping, page-tracked chunks.

    Sentences are packed greedily up to chunk_size characters; each new chunk
    re-starts with the tail sentences of the previous one so answers spanning
    a boundary remain retrievable. Page provenance is preserved so citations
    can point at document + page.

    Args:
        doc: Normalized document.
        chunk_size: Target chunk size in characters.
        overlap: Approximate character overlap between adjacent chunks.

    Returns:
        Ordered list of chunks.
    """
    chunks: list[Chunk] = []
    seq = 0

    for page_number, page_text in enumerate(doc.pages, start=1):
        sentences = _split_sentences(page_text)
        if not sentences:
            continue

        current: list[str] = []
        current_len = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            current.append(sentence)
            current_len += len(sentence) + 1
            i += 1

            if current_len >= chunk_size or i == len(sentences):
                text = " ".join(current).strip()
                if text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}:{page_number}:{seq}",
                            doc_id=doc.doc_id,
                            source=doc.source,
                            source_type=doc.source_type,
                            page=page_number,
                            seq=seq,
                            text=text,
                            created_at=doc.created_at,
                        )
                    )
                    seq += 1

                if i == len(sentences):
                    break

                # Carry tail sentences forward as overlap
                tail: list[str] = []
                tail_len = 0
                for sent in reversed(current):
                    if tail_len + len(sent) > overlap:
                        break
                    tail.insert(0, sent)
                    tail_len += len(sent) + 1
                current = tail
                current_len = tail_len

    return chunks
