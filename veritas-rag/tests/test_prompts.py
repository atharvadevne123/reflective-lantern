"""Tests for veritas-rag generation prompts module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generation.prompts import CONSTRAINED_SYSTEM_PROMPT, INSUFFICIENT_EVIDENCE_MESSAGE


def test_constrained_system_prompt_is_string():
    assert isinstance(CONSTRAINED_SYSTEM_PROMPT, str)


def test_constrained_system_prompt_not_empty():
    assert len(CONSTRAINED_SYSTEM_PROMPT.strip()) > 0


def test_constrained_prompt_mentions_evidence():
    assert "evidence" in CONSTRAINED_SYSTEM_PROMPT.lower()


def test_constrained_prompt_mentions_excerpts():
    assert "excerpt" in CONSTRAINED_SYSTEM_PROMPT.lower()


def test_insufficient_evidence_message_is_string():
    assert isinstance(INSUFFICIENT_EVIDENCE_MESSAGE, str)


def test_insufficient_evidence_message_not_empty():
    assert len(INSUFFICIENT_EVIDENCE_MESSAGE.strip()) > 0


def test_insufficient_evidence_message_mentions_evidence():
    assert "evidence" in INSUFFICIENT_EVIDENCE_MESSAGE.lower()


def test_insufficient_evidence_message_mentions_insufficient():
    assert "insufficient" in INSUFFICIENT_EVIDENCE_MESSAGE.lower()


def test_system_prompt_contains_numbered_rules():
    assert "1." in CONSTRAINED_SYSTEM_PROMPT


def test_system_prompt_mentions_prior_knowledge():
    assert (
        "prior" in CONSTRAINED_SYSTEM_PROMPT.lower()
        or "knowledge" in CONSTRAINED_SYSTEM_PROMPT.lower()
    )
