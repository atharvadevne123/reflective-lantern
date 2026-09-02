"""Stages 2-3 tests: BM25, embeddings, ANN, fusion, reranking."""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.ann import VectorIndex
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.embedder import HashingEmbedder
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.rerank import LexicalReranker


class TestBM25:
    def test_tokenize_strips_stopwords(self):
        assert "the" not in tokenize("the quick fox")
        assert tokenize("Quick FOX!") == ["quick", "fox"]

    def test_exact_term_match_ranks_first(self):
        index = BM25Index()
        index.add("c1", "solar rebate program for households")
        index.add("c2", "battery storage incentive details")
        index.add("c3", "grid connection export limits")
        results = index.search("solar rebate")
        assert results[0][0] == "c1"

    def test_remove_drops_chunk(self):
        index = BM25Index()
        index.add("c1", "unique zebra content")
        index.remove("c1")
        assert index.search("zebra") == []
        assert len(index) == 0

    def test_empty_query_returns_nothing(self):
        index = BM25Index()
        index.add("c1", "text")
        assert index.search("the of and") == []

    def test_top_k_respected(self):
        index = BM25Index()
        for i in range(20):
            index.add(f"c{i}", f"solar document number {i}")
        assert len(index.search("solar", top_k=5)) == 5


class TestEmbedder:
    def test_deterministic(self):
        emb = HashingEmbedder(dim=64)
        a = emb.embed(["hello world"])
        b = emb.embed(["hello world"])
        np.testing.assert_array_equal(a, b)

    def test_normalized(self):
        emb = HashingEmbedder(dim=64)
        vecs = emb.embed(["some text here", "other content entirely"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_similar_texts_closer_than_dissimilar(self):
        emb = HashingEmbedder(dim=384)
        vecs = emb.embed(
            [
                "solar rebate program for rooftop panels",
                "rooftop solar panel rebate scheme",
                "recipe for chocolate cake with frosting",
            ]
        )
        sim_related = float(vecs[0] @ vecs[1])
        sim_unrelated = float(vecs[0] @ vecs[2])
        assert sim_related > sim_unrelated


class TestVectorIndex:
    def test_nearest_neighbour_found(self):
        emb = HashingEmbedder(dim=128)
        texts = ["solar rebates", "battery storage", "grid limits"]
        index = VectorIndex(dim=128)
        index.add(["c1", "c2", "c3"], emb.embed(texts))
        query = emb.embed(["solar rebate program"])[0]
        results = index.search(query, top_k=1)
        assert results[0][0] == "c1"

    def test_deleted_ids_filtered(self):
        emb = HashingEmbedder(dim=128)
        index = VectorIndex(dim=128)
        index.add(["c1", "c2"], emb.embed(["alpha text", "beta text"]))
        index.remove(["c1"])
        results = index.search(emb.embed(["alpha text"])[0], top_k=5)
        assert all(cid != "c1" for cid, _ in results)

    def test_ivf_path_over_threshold(self):
        emb = HashingEmbedder(dim=32)
        n = 600
        texts = [f"document about topic {i} with words {i % 7} {i % 13}" for i in range(n)]
        index = VectorIndex(dim=32, ivf_threshold=500)
        index.add([f"c{i}" for i in range(n)], emb.embed(texts))
        results = index.search(emb.embed(["document about topic 5"])[0], top_k=10)
        assert len(results) == 10

    def test_empty_index(self):
        index = VectorIndex(dim=8)
        assert index.search(np.zeros(8, dtype=np.float32)) == []


class TestFusion:
    def test_agreement_ranks_highest(self):
        bm25 = [("shared", 5.0), ("bm_only", 4.0)]
        vec = [("shared", 0.9), ("vec_only", 0.8)]
        fused = reciprocal_rank_fusion(bm25, vec)
        assert fused[0][0] == "shared"

    def test_ranks_recorded(self):
        fused = reciprocal_rank_fusion([("a", 1.0)], [("b", 1.0)])
        by_id = {cid: (b, v) for cid, _, b, v in fused}
        assert by_id["a"] == (1, None)
        assert by_id["b"] == (None, 1)

    def test_empty_inputs(self):
        assert reciprocal_rank_fusion([], []) == []


class TestReranker:
    def test_relevant_text_scores_higher(self):
        rr = LexicalReranker()
        scores = rr.score(
            "solar rebate amount",
            [
                "The solar rebate amount is 900 dollars per kilowatt.",
                "Battery warranties must cover ten years.",
            ],
        )
        assert scores[0] > scores[1]

    def test_scores_bounded(self):
        rr = LexicalReranker()
        scores = rr.score("query terms", ["query terms appear here", "nothing relevant"])
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_proximity_rewarded(self):
        rr = LexicalReranker()
        tight = "export limit five kilowatts single phase"
        loose = (
            "export rules are long. many words pass. the limit topic changes. "
            "eventually five appears. kilowatts mentioned last."
        )
        scores = rr.score("export limit five kilowatts", [tight, loose])
        assert scores[0] > scores[1]

    @pytest.mark.parametrize("query", ["", "the of and"])
    def test_stopword_only_query(self, query):
        rr = LexicalReranker()
        assert rr.score(query, ["some text"]) == [0.0]


class TestPipelineRetrieval:
    def test_hybrid_finds_expected_document(self, pipeline):
        answer = pipeline.query("How much is the rooftop solar rebate per kilowatt?")
        assert answer.answered
        assert any("solar-policy" in c.source for c in answer.citations)

    def test_semantic_match_without_exact_terms(self, pipeline):
        # "power storage" is not verbatim in the corpus; "battery storage" is
        answer = pipeline.query("What incentive exists for home power storage batteries?")
        assert answer.answered
        assert any("battery-guide" in c.source for c in answer.citations)


class TestBM25Parametrized:
    @pytest.mark.parametrize("stopword", ["the", "of", "and", "a", "is"])
    def test_common_stopwords_stripped(self, stopword: str) -> None:
        """Common English stopwords are removed during tokenization."""
        assert stopword not in tokenize(f"this {stopword} that")

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_top_k_limits_results(self, top_k: int) -> None:
        """BM25Index.search respects the top_k limit."""
        index = BM25Index()
        for i in range(10):
            index.add(f"doc{i}", f"solar energy content document {i}")
        results = index.search("solar", top_k=top_k)
        assert len(results) == top_k


class TestEmbedderParametrized:
    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_embedding_dimension(self, dim: int) -> None:
        """HashingEmbedder produces vectors of the requested dimension."""
        emb = HashingEmbedder(dim=dim)
        vecs = emb.embed(["test text"])
        assert vecs.shape == (1, dim)

    @pytest.mark.parametrize("n_texts", [1, 5, 10])
    def test_batch_embedding_shape(self, n_texts: int) -> None:
        """HashingEmbedder handles batches of various sizes."""
        emb = HashingEmbedder(dim=64)
        texts = [f"document number {i}" for i in range(n_texts)]
        vecs = emb.embed(texts)
        assert vecs.shape[0] == n_texts
