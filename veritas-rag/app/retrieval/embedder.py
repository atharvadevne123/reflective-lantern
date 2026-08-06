"""Deterministic hashed-feature embedder with a pluggable interface.

The default embedder maps word unigrams/bigrams and character trigrams into a
fixed-dimension space via feature hashing, then L2-normalizes. It is fully
deterministic, dependency-free, and fast — suitable for tests, CI, and dev.

For production semantic quality, implement `Embedder` with a neural model
(e.g. sentence-transformers or a hosted embedding API) — the rest of the
pipeline is agnostic to the backend.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Protocol

import numpy as np

from .bm25 import tokenize


class Embedder(Protocol):
    """Interface every embedding backend must satisfy."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized embeddings."""
        ...


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = struct.unpack("<Q", digest)[0]
    index = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbedder:
    """Feature-hashing embedder (deterministic, no model download)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        features = [f"w:{t}" for t in tokens]
        features += [f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        compact = "".join(tokens)
        features += [f"c:{compact[i : i + 3]}" for i in range(max(0, len(compact) - 2))]
        return features

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts into L2-normalized hashed-feature vectors."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                index, sign = _bucket(feature, self.dim)
                out[row, index] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return out / norms
