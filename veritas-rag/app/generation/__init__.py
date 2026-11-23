"""Stages 5-7 — constrained generation, citations, hallucination fallback."""

from .extractive import ExtractiveGenerator
from .fallback import GateDecision, evaluate_evidence
from .prompts import CONSTRAINED_SYSTEM_PROMPT, INSUFFICIENT_EVIDENCE_MESSAGE

__all__ = [
    "CONSTRAINED_SYSTEM_PROMPT",
    "ExtractiveGenerator",
    "GateDecision",
    "INSUFFICIENT_EVIDENCE_MESSAGE",
    "evaluate_evidence",
]
