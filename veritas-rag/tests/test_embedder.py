"""Tests for veritas-rag hashing embedder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retrieval.embedder import HashingEmbedder


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=64)


def test_embed_returns_array(embedder):
    result = embedder.embed(["hello world"])
    assert isinstance(result, np.ndarray)


def test_embed_shape(embedder):
    result = embedder.embed(["a", "b", "c"])
    assert result.shape == (3, 64)


def test_embed_dtype_float32(embedder):
    result = embedder.embed(["test"])
    assert result.dtype == np.float32


def test_embed_deterministic(embedder):
    r1 = embedder.embed(["same text"])
    r2 = embedder.embed(["same text"])
    np.testing.assert_array_equal(r1, r2)


def test_embed_different_texts_differ(embedder):
    r1 = embedder.embed(["dogs are great"])
    r2 = embedder.embed(["cats are great"])
    assert not np.allclose(r1, r2)


def test_embed_unit_norm(embedder):
    result = embedder.embed(["unit norm check"])
    norms = np.linalg.norm(result, axis=1)
    assert all(abs(n - 1.0) < 1e-5 for n in norms)


def test_embed_empty_list(embedder):
    result = embedder.embed([])
    assert result.shape == (0, 64)


def test_dim_attribute(embedder):
    assert embedder.dim == 64


def test_different_dims_produce_different_shapes():
    e32 = HashingEmbedder(dim=32)
    e128 = HashingEmbedder(dim=128)
    r32 = e32.embed(["test"])
    r128 = e128.embed(["test"])
    assert r32.shape == (1, 32)
    assert r128.shape == (1, 128)


@pytest.mark.parametrize("text", ["hello", "world", "foo bar baz", "longer text"])
def test_single_text_embed_has_unit_norm(text):
    emb = HashingEmbedder(dim=64)
    result = emb.embed([text])
    norm = float(np.linalg.norm(result[0]))
    assert abs(norm - 1.0) < 1e-5


def test_similar_texts_higher_cosine_than_dissimilar():
    emb = HashingEmbedder(dim=256)
    v1 = emb.embed(["python machine learning models"])[0]
    v2 = emb.embed(["python machine learning neural"])[0]
    v3 = emb.embed(["cooking pasta italian recipe"])[0]
    sim_similar = float(v1 @ v2)
    sim_dissimilar = float(v1 @ v3)
    assert sim_similar > sim_dissimilar


def test_embed_batch_returns_correct_shape():
    emb = HashingEmbedder(dim=128)
    texts = ["a", "b", "c", "d"]
    result = emb.embed(texts)
    assert result.shape == (4, 128)


def test_embed_empty_batch_returns_zero_rows():
    emb = HashingEmbedder(dim=64)
    result = emb.embed([])
    assert result.shape == (0, 64)


def test_embed_dtype_is_float32():
    emb = HashingEmbedder(dim=64)
    result = emb.embed(["hello world"])
    assert result.dtype == np.float32


def test_embed_dim_attribute_matches_output():
    emb = HashingEmbedder(dim=192)
    assert emb.dim == 192
    result = emb.embed(["test"])
    assert result.shape[1] == 192


def test_embed_all_rows_unit_norm():
    emb = HashingEmbedder(dim=128)
    texts = ["hello world", "machine learning", "quantum computing"]
    result = emb.embed(texts)
    for row in result:
        norm = float(np.linalg.norm(row))
        assert abs(norm - 1.0) < 1e-4


def test_embed_deterministic_across_calls():
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed(["same text exactly"])
    v2 = emb.embed(["same text exactly"])
    assert np.allclose(v1, v2)


def test_embed_different_text_different_vectors():
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed(["alpha beta gamma"])
    v2 = emb.embed(["delta epsilon zeta"])
    assert not np.allclose(v1, v2)


@pytest.mark.parametrize("dim", [32, 64, 128, 256])
def test_embed_various_dims(dim):
    emb = HashingEmbedder(dim=dim)
    result = emb.embed(["test embedding dimension"])
    assert result.shape == (1, dim)
