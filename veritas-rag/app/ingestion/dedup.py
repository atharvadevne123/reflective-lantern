"""Exact and near-duplicate detection.

Exact duplicates are caught with a SHA-256 content hash. Near-duplicates
(boilerplate variants, forwarded emails, re-exported PDFs) are caught with
MinHash signatures over word shingles — at 10M-document scale the signatures
are bucketed with locality-sensitive hashing so candidate pairs are found
without O(n^2) comparisons.
"""

from __future__ import annotations

import hashlib
import struct

_NUM_HASHES = 64
_SHINGLE_SIZE = 3
_LSH_BANDS = 16  # 16 bands x 4 rows = 64 hashes


def content_hash(text: str) -> str:
    """SHA-256 hash of normalized text (exact-duplicate key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shingles(text: str, size: int = _SHINGLE_SIZE) -> set[str]:
    """Return the set of word-level n-grams of length *size* from *text*."""
    words = text.lower().split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def _hash64(value: str, seed: int) -> int:
    """Hash *value* to a 64-bit integer using BLAKE2b keyed by *seed*."""
    digest = hashlib.blake2b(
        value.encode("utf-8"), digest_size=8, salt=struct.pack("<Q", seed)[:8]
    ).digest()
    return struct.unpack("<Q", digest)[0]


def minhash_signature(text: str) -> tuple[int, ...]:
    """Deterministic 64-value MinHash signature of a text's word shingles."""
    shingles = _shingles(text)
    if not shingles:
        return tuple([0] * _NUM_HASHES)
    return tuple(min(_hash64(s, seed) for s in shingles) for seed in range(_NUM_HASHES))


def signature_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity between two MinHash signatures."""
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return matches / len(sig_a)


class DuplicateDetector:
    """Streaming duplicate detector with exact + LSH-bucketed near-dup checks."""

    def __init__(self, near_threshold: float = 0.85) -> None:
        """Initialise the detector with a Jaccard similarity threshold for near-duplicates."""
        self.near_threshold = near_threshold
        self._exact: dict[str, str] = {}  # content_hash -> doc_id
        self._signatures: dict[str, tuple[int, ...]] = {}  # doc_id -> signature
        self._lsh_buckets: dict[tuple[int, int], list[str]] = {}

    @staticmethod
    def _bands(signature: tuple[int, ...]) -> list[tuple[int, int]]:
        """Split *signature* into LSH band keys for bucket lookup."""
        rows = _NUM_HASHES // _LSH_BANDS
        keys = []
        for band in range(_LSH_BANDS):
            chunk = signature[band * rows : (band + 1) * rows]
            band_hash = hash(chunk)
            keys.append((band, band_hash))
        return keys

    def check(self, text: str) -> tuple[str | None, float]:
        """Return (duplicate_of_doc_id, similarity) or (None, 0.0).

        Args:
            text: Normalized document text.
        """
        digest = content_hash(text)
        if digest in self._exact:
            return self._exact[digest], 1.0

        signature = minhash_signature(text)
        candidates: set[str] = set()
        for key in self._bands(signature):
            candidates.update(self._lsh_buckets.get(key, ()))

        best_id: str | None = None
        best_sim = 0.0
        for doc_id in candidates:
            sim = signature_similarity(signature, self._signatures[doc_id])
            if sim > best_sim:
                best_id, best_sim = doc_id, sim

        if best_id is not None and best_sim >= self.near_threshold:
            return best_id, best_sim
        return None, 0.0

    def register(self, doc_id: str, text: str) -> None:
        """Add a document to the duplicate index."""
        digest = content_hash(text)
        self._exact[digest] = doc_id
        signature = minhash_signature(text)
        self._signatures[doc_id] = signature
        for key in self._bands(signature):
            self._lsh_buckets.setdefault(key, []).append(doc_id)

    def __len__(self) -> int:
        """Return the number of documents registered in the index."""
        return len(self._signatures)
