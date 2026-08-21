"""Tests for model training, inference, and FAISS index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import INTERACTION_COLS, ITEM_COLS, USER_COLS

_FEATURE_COLS = USER_COLS + ITEM_COLS + INTERACTION_COLS


class TestModelTraining:
    def test_train_returns_pipeline_and_metrics(self, feature_df, binary_labels) -> None:
        from app.model import train_model

        pipe, metrics = train_model(feature_df, binary_labels)
        assert pipe is not None
        assert "auc_mean" in metrics

    def test_auc_in_valid_range(self, feature_df, binary_labels) -> None:
        from app.model import train_model

        _, metrics = train_model(feature_df, binary_labels)
        assert 0.0 <= metrics["auc_mean"] <= 1.0

    def test_metrics_contains_required_keys(self, feature_df, binary_labels) -> None:
        from app.model import train_model

        _, metrics = train_model(feature_df, binary_labels)
        for key in ("auc_mean", "auc_std", "n_features", "n_samples"):
            assert key in metrics

    def test_n_features_matches_input(self, feature_df, binary_labels) -> None:
        from app.model import train_model

        _, metrics = train_model(feature_df, binary_labels)
        assert metrics["n_features"] == len(_FEATURE_COLS)

    def test_model_file_created(self, trained_pipeline, tmp_path, monkeypatch) -> None:

        out = tmp_path / "test_model.joblib"
        monkeypatch.setattr("app.model.MODEL_PATH", out)
        from app.features import make_sample_dataframe
        from app.model import train_model

        df = make_sample_dataframe(n=50)
        X = df[_FEATURE_COLS]
        y = pd.Series(np.random.randint(0, 2, 50), name="purchased")
        train_model(X, y)
        assert out.exists()


class TestModelInference:
    def test_predict_returns_proba_and_labels(self, trained_pipeline, feature_df) -> None:
        from app.model import predict_intent

        probas, labels = predict_intent(trained_pipeline, feature_df)
        assert len(probas) == len(feature_df)
        assert len(labels) == len(feature_df)

    def test_probas_in_range(self, trained_pipeline, feature_df) -> None:
        from app.model import predict_intent

        probas, _ = predict_intent(trained_pipeline, feature_df)
        for p in probas:
            assert 0.0 <= p <= 1.0

    def test_labels_are_binary(self, trained_pipeline, feature_df) -> None:
        from app.model import predict_intent

        _, labels = predict_intent(trained_pipeline, feature_df)
        assert all(label in (0, 1) for label in labels)

    @pytest.mark.parametrize("n_rows", [1, 5, 20])
    def test_predict_various_batch_sizes(self, trained_pipeline, n_rows) -> None:
        from app.features import make_sample_dataframe
        from app.model import predict_intent

        df = make_sample_dataframe(n=n_rows)
        X = df[_FEATURE_COLS]
        probas, labels = predict_intent(trained_pipeline, X)
        assert len(probas) == n_rows
        assert len(labels) == n_rows


class TestFaissIndex:
    def test_build_and_search_returns_results(self) -> None:
        from app.model import build_faiss_index, search_similar_items

        vecs = np.random.rand(50, 16).astype(np.float32)
        ids = [f"item_{i}" for i in range(50)]
        index = build_faiss_index(vecs, ids)
        query = np.random.rand(16).astype(np.float32)
        results = search_similar_items(index, ids, query, top_k=3)
        assert len(results) == 3

    def test_similarity_scores_positive(self) -> None:
        from app.model import build_faiss_index, search_similar_items

        vecs = np.random.rand(30, 8).astype(np.float32)
        ids = [f"item_{i}" for i in range(30)]
        index = build_faiss_index(vecs, ids)
        query = np.random.rand(8).astype(np.float32)
        results = search_similar_items(index, ids, query, top_k=5)
        for r in results:
            assert r["similarity_score"] >= 0.0

    def test_brute_force_fallback(self) -> None:
        from app.model import _BruteForceIndex, search_similar_items

        vecs = np.random.rand(20, 8).astype(np.float32)
        ids = [f"item_{i}" for i in range(20)]
        index = _BruteForceIndex(vecs, ids)
        query = np.random.rand(8).astype(np.float32)
        results = search_similar_items(index, ids, query, top_k=3)
        assert len(results) == 3
