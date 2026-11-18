"""Extended tests for veritas-rag dedup: content_hash, minhash, and similarity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.dedup import (
    DuplicateDetector,
    content_hash,
    minhash_signature,
    signature_similarity,
)


def test_content_hash_returns_string():
    result = content_hash("hello world")
    assert isinstance(result, str)


def test_content_hash_is_64_chars():
    result = content_hash("test")
    assert len(result) == 64


def test_content_hash_deterministic():
    h1 = content_hash("energy consumption model")
    h2 = content_hash("energy consumption model")
    assert h1 == h2


def test_content_hash_different_texts_differ():
    h1 = content_hash("text one")
    h2 = content_hash("text two")
    assert h1 != h2


def test_content_hash_empty_string():
    result = content_hash("")
    assert isinstance(result, str)
    assert len(result) == 64


def test_minhash_signature_returns_tuple():
    result = minhash_signature("hello world foo bar baz")
    assert isinstance(result, tuple)


def test_minhash_signature_length():
    result = minhash_signature("energy consumption model prediction")
    assert len(result) == 64


def test_minhash_signature_deterministic():
    s1 = minhash_signature("the quick brown fox")
    s2 = minhash_signature("the quick brown fox")
    assert s1 == s2


def test_minhash_signature_empty_text():
    result = minhash_signature("")
    assert isinstance(result, tuple)


def test_signature_similarity_identical_texts():
    text = "energy consumption prediction model accuracy"
    s1 = minhash_signature(text)
    s2 = minhash_signature(text)
    sim = signature_similarity(s1, s2)
    assert sim == pytest.approx(1.0)


def test_signature_similarity_different_texts():
    s1 = minhash_signature("energy consumption model")
    s2 = minhash_signature("quantum physics nuclear reaction")
    sim = signature_similarity(s1, s2)
    assert 0.0 <= sim <= 1.0


def test_signature_similarity_similar_texts_higher():
    s1 = minhash_signature("energy model prediction accuracy high")
    s2 = minhash_signature("energy model prediction accuracy medium")
    s3 = minhash_signature("quantum physics nuclear fusion plasma")
    sim_similar = signature_similarity(s1, s2)
    sim_different = signature_similarity(s1, s3)
    assert sim_similar >= sim_different


def test_duplicate_detector_new_doc_not_duplicate():
    dd = DuplicateDetector()
    result = dd.is_duplicate("doc1", "Energy consumption dropped this month.")
    assert result is False


def test_duplicate_detector_same_doc_is_duplicate():
    dd = DuplicateDetector()
    text = "Energy consumption dropped this month in the building."
    dd.is_duplicate("doc1", text)
    result = dd.is_duplicate("doc2", text)
    assert result is True


@pytest.mark.parametrize(
    "text",
    [
        "short text here",
        "a" * 500,
        "energy model prediction accuracy measurement threshold",
    ],
)
def test_minhash_signature_various_lengths(text):
    result = minhash_signature(text)
    assert len(result) == 64
