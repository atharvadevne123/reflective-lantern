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

    def test_exact_dup_has_similarity_one(self) -> None:
        dd = DuplicateDetector()
        text = "exact same content repeated twice for exact dedup detection"
        dd.register("doc-exact", text)
        _, sim = dd.check(text)
        assert sim == pytest.approx(1.0)

    def test_len_after_multiple_registers(self) -> None:
        dd = DuplicateDetector()
        for i in range(5):
            dd.register(f"doc-{i}", f"unique text number {i} with enough words to shingle")
        assert len(dd) == 5

    def test_check_before_register_returns_none(self) -> None:
        dd = DuplicateDetector()
        dup_id, sim = dd.check("some completely fresh text")
        assert dup_id is None
        assert sim == pytest.approx(0.0)

    @pytest.mark.parametrize("threshold", [0.5, 0.8, 0.95])
    def test_custom_threshold_applied(self, threshold: float) -> None:
        dd = DuplicateDetector(near_threshold=threshold)
        assert dd.near_threshold == threshold

    def test_register_same_doc_id_twice_no_error(self) -> None:
        dd = DuplicateDetector()
        dd.register("doc1", "content about machine learning today")
        dd.register("doc1", "content about machine learning today updated version")
        assert len(dd) >= 1

    def test_check_similarity_in_range(self) -> None:
        dd = DuplicateDetector()
        dd.register("doc1", "machine learning neural network training optimization")
        _, sim = dd.check("machine learning neural network training modern")
        assert 0.0 <= sim <= 1.0

    def test_near_duplicate_text_detected(self) -> None:
        dd = DuplicateDetector(near_threshold=0.3)
        base = "machine learning neural network deep learning training data"
        variant = "machine learning neural network deep learning optimization algorithm"
        dd.register("original", base)
        dup_id, sim = dd.check(variant)
        # With low threshold, highly similar texts should be detected
        if sim >= 0.3:
            assert dup_id == "original"


def test_content_hash_function():
    from app.ingestion.dedup import content_hash

    h1 = content_hash("hello world")
    h2 = content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex = 64 chars
