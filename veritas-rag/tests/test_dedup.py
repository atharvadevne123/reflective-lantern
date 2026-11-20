"""Tests for veritas-rag app/ingestion/dedup.py."""

from __future__ import annotations

import pytest
from app.ingestion.dedup import (
    DuplicateDetector,
    content_hash,
    minhash_signature,
    signature_similarity,
)


class TestContentHash:
    def test_returns_string(self) -> None:
        assert isinstance(content_hash("hello"), str)

    def test_deterministic(self) -> None:
        assert content_hash("same text") == content_hash("same text")

    def test_different_texts_differ(self) -> None:
        assert content_hash("text a") != content_hash("text b")

    def test_fixed_length_sha256(self) -> None:
        assert len(content_hash("anything")) == 64


class TestMinhashSignature:
    def test_returns_tuple(self) -> None:
        sig = minhash_signature("hello world test")
        assert isinstance(sig, tuple)

    def test_correct_length(self) -> None:
        sig = minhash_signature("some text here")
        assert len(sig) == 64

    def test_empty_string_returns_zeros(self) -> None:
        sig = minhash_signature("")
        assert all(v == 0 for v in sig)

    def test_deterministic(self) -> None:
        text = "identical text here today"
        assert minhash_signature(text) == minhash_signature(text)

    def test_different_texts_differ(self) -> None:
        s1 = minhash_signature("alpha beta gamma delta")
        s2 = minhash_signature("one two three four five")
        assert s1 != s2


class TestSignatureSimilarity:
    def test_identical_signatures_score_one(self) -> None:
        sig = minhash_signature("hello world today")
        assert signature_similarity(sig, sig) == pytest.approx(1.0)

    def test_empty_signatures_score_zero(self) -> None:
        assert signature_similarity((), ()) == pytest.approx(0.0)

    def test_result_in_range(self) -> None:
        s1 = minhash_signature("alpha beta gamma")
        s2 = minhash_signature("alpha gamma delta epsilon")
        score = signature_similarity(s1, s2)
        assert 0.0 <= score <= 1.0


class TestDuplicateDetector:
    def test_empty_detector_len_zero(self) -> None:
        dd = DuplicateDetector()
        assert len(dd) == 0

    def test_check_returns_none_on_empty(self) -> None:
        dd = DuplicateDetector()
        dup_id, sim = dd.check("unique content here today yes")
        assert dup_id is None

    def test_exact_duplicate_detected(self) -> None:
        dd = DuplicateDetector()
        text = "exactly the same text right here"
        dd.register("doc1", text)
        dup_id, sim = dd.check(text)
        assert dup_id == "doc1"
        assert sim == pytest.approx(1.0)

    def test_register_increments_len(self) -> None:
        dd = DuplicateDetector()
        dd.register("doc1", "some unique content here today indeed")
        assert len(dd) == 1

    def test_different_text_no_dup(self) -> None:
        dd = DuplicateDetector()
        dd.register("doc1", "paris is the capital of france europe")
        dup_id, sim = dd.check("completely unrelated content about robotics")
        assert dup_id is None

    def test_check_returns_tuple(self) -> None:
        dd = DuplicateDetector()
        result = dd.check("some text")
        assert isinstance(result, tuple)
        assert len(result) == 2
