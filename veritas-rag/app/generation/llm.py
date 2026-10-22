"""Anthropic-backed constrained generation with native citations.

Each retrieved chunk is passed as a `document` content block with citations
enabled, so the API returns claim-level `char_location` citations that map
answer spans back to specific evidence — the model cannot silently attribute
a claim to nothing.
"""

from __future__ import annotations

import logging
import time

from ..schemas import Citation, ScoredChunk
from .prompts import CONSTRAINED_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY_S = 1.0


class AnthropicGenerator:
    """Constrained abstractive answering via the Anthropic Messages API."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, chunks: list[ScoredChunk]) -> tuple[str, list[Citation]]:
        """Ask the model for an evidence-bound answer with citations.

        Args:
            query: The user question.
            chunks: Trust-scored supporting chunks, best first.

        Returns:
            (answer_text, citations). An empty answer signals the model
            declared INSUFFICIENT EVIDENCE (or refused), which the fallback
            stage converts into the standard insufficient-evidence response.
        """
        content: list[dict] = [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": scored.chunk.text,
                },
                "title": f"{scored.chunk.source} (page {scored.chunk.page})",
                "citations": {"enabled": True},
            }
            for scored in chunks
        ]
        content.append({"type": "text", "text": query})

        last_exc: Exception | None = None
        response = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=CONSTRAINED_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Anthropic API error on attempt %d/%d: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_S * (2**attempt))

        if response is None:
            logger.error("All %d Anthropic API attempts failed: %s", _MAX_RETRIES + 1, last_exc)
            return "", []

        if response.stop_reason == "refusal":
            logger.warning("Model refused the generation request")
            return "", []

        answer_parts: list[str] = []
        citations: list[Citation] = []
        for block in response.content:
            if block.type != "text":
                continue
            answer_parts.append(block.text)
            for cite in getattr(block, "citations", None) or []:
                doc_index = getattr(cite, "document_index", None)
                if doc_index is None or doc_index >= len(chunks):
                    continue
                scored = chunks[doc_index]
                citations.append(
                    Citation(
                        claim=block.text.strip(),
                        chunk_id=scored.chunk.chunk_id,
                        doc_id=scored.chunk.doc_id,
                        source=scored.chunk.source,
                        page=scored.chunk.page,
                        quote=getattr(cite, "cited_text", ""),
                    )
                )

        answer = "".join(answer_parts).strip()
        if "INSUFFICIENT EVIDENCE" in answer.upper():
            return "", []
        return answer, citations
