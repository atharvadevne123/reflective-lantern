"""Stage 7 — the hallucination fallback gate.

An answer is only released when the evidence clears the confidence bar.
Anything below it — weak retrieval, stale sources, retriever disagreement,
or a generator that produced no grounded sentences — becomes an explicit
"insufficient evidence" response instead of a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schemas import ScoredChunk
from ..scoring.trust import answer_confidence

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Outcome of the evidence gate.

    Attributes:
        allowed: True when generation may proceed / an answer may be released.
        confidence: Mean trust of the supporting chunks.
        supporting_chunks: Chunks whose individual trust cleared the floor.
        reason: Machine-readable explanation for the trace.
    """

    allowed: bool
    confidence: float
    supporting_chunks: list[ScoredChunk]
    reason: str


def evaluate_evidence(
    chunks: list[ScoredChunk],
    confidence_threshold: float = 0.45,
    chunk_trust_floor: float = 0.35,
    min_supporting_chunks: int = 1,
) -> GateDecision:
    """Decide whether the retrieved evidence is strong enough to answer.

    Args:
        chunks: Trust-scored chunks from the pipeline.
        confidence_threshold: Minimum mean trust across supporting chunks.
        chunk_trust_floor: A chunk only counts as support above this trust.
        min_supporting_chunks: How many supporting chunks are required.
    """
    if not chunks:
        logger.debug("Evidence gate: no chunks retrieved")
        return GateDecision(False, 0.0, [], "no_chunks_retrieved")

    supporting = [c for c in chunks if c.trust >= chunk_trust_floor]
    if len(supporting) < min_supporting_chunks:
        logger.debug(
            "Evidence gate: only %d supporting chunk(s), need %d",
            len(supporting),
            min_supporting_chunks,
        )
        return GateDecision(
            False,
            answer_confidence(chunks),
            supporting,
            "too_few_supporting_chunks",
        )

    confidence = answer_confidence(supporting)
    if confidence < confidence_threshold:
        logger.debug(
            "Evidence gate: confidence %.4f below threshold %.4f",
            confidence,
            confidence_threshold,
        )
        return GateDecision(False, confidence, supporting, "confidence_below_threshold")

    logger.debug(
        "Evidence gate: passed with confidence=%.4f, %d chunks", confidence, len(supporting)
    )
    return GateDecision(True, confidence, supporting, "evidence_sufficient")
