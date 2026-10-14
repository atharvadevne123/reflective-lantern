"""Tests for model training and prediction in Social-Pulse."""

from __future__ import annotations

import numpy as np
import pytest

from app.model import (
    PredictResult,
    TrainResult,
    _to_dense,
    generate_synthetic_data,
    predict,
    train,
)


class TestGenerateSyntheticData:
    def test_returns_correct_count(self) -> None:
        samples, labels = generate_synthetic_data(n=50)
        assert len(samples) == 50
        assert len(labels) == 50

    def test_labels_are_binary(self) -> None:
        _, labels = generate_synthetic_data(n=100)
        assert all(lbl in (0, 1) for lbl in labels)

    def test_samples_have_required_keys(self) -> None:
        samples, _ = generate_synthetic_data(n=10)
        required = {"text", "author_followers", "has_media", "has_link",
                    "hashtag_count", "mention_count", "hour_of_day", "day_of_week"}
        for s in samples:
            assert required.issubset(s.keys())

    def test_viral_rate_approximately_25_pct(self) -> None:
        _, labels = generate_synthetic_data(n=1000)
        rate = sum(labels) / len(labels)
        assert 0.10 <= rate <= 0.40  # 25% ± wide tolerance

    def test_text_within_280_chars(self) -> None:
        samples, _ = generate_synthetic_data(n=50)
        for s in samples:
            assert len(s["text"]) <= 280

    def test_hour_of_day_in_range(self) -> None:
        samples, _ = generate_synthetic_data(n=50)
        for s in samples:
            assert 0 <= s["hour_of_day"] <= 23


class TestTrainModel:
    @pytest.fixture(scope="class")
    def trained(self):
        samples, labels = generate_synthetic_data(n=200)
        return train(samples, labels, n_folds=2, model_version="test-0.1")

    def test_returns_train_result(self, trained: TrainResult) -> None:
        assert isinstance(trained, TrainResult)

    def test_auc_roc_above_chance(self, trained: TrainResult) -> None:
        assert trained.auc_roc >= 0.5

    def test_f1_non_negative(self, trained: TrainResult) -> None:
        assert trained.f1 >= 0.0

    def test_cv_scores_populated(self, trained: TrainResult) -> None:
        assert len(trained.cv_auc_scores) == 2

    def test_train_samples_count_correct(self, trained: TrainResult) -> None:
        assert trained.train_samples == 200

    def test_model_version_preserved(self, trained: TrainResult) -> None:
        assert trained.model_version == "test-0.1"

    def test_duration_positive(self, trained: TrainResult) -> None:
        assert trained.duration_sec > 0.0

    def test_model_files_written(self) -> None:
        from app.model import CURRENT_MODEL_PATH, CURRENT_PIPELINE_PATH
        assert CURRENT_MODEL_PATH.exists()
        assert CURRENT_PIPELINE_PATH.exists()


class TestPredictModel:
    @pytest.fixture(scope="class")
    def model_artifacts(self):
        samples, labels = generate_synthetic_data(n=200)
        train(samples, labels, n_folds=2)
        from app.model import load_model
        result = load_model()
        assert result is not None
        return result

    def test_predict_returns_predict_result(self, model_artifacts, sample_post: dict) -> None:
        pipeline, model = model_artifacts
        result = predict(sample_post, pipeline, model)
        assert isinstance(result, PredictResult)

    def test_probability_in_unit_interval(self, model_artifacts, sample_post: dict) -> None:
        pipeline, model = model_artifacts
        result = predict(sample_post, pipeline, model)
        assert 0.0 <= result.viral_probability <= 1.0

    def test_label_consistent_with_probability(self, model_artifacts, sample_post: dict) -> None:
        pipeline, model = model_artifacts
        result = predict(sample_post, pipeline, model)
        expected_label = result.viral_probability >= 0.5
        assert result.viral_label == expected_label

    def test_confidence_in_unit_interval(self, model_artifacts, sample_post: dict) -> None:
        pipeline, model = model_artifacts
        result = predict(sample_post, pipeline, model)
        assert 0.0 <= result.ensemble_confidence <= 1.0

    def test_latency_positive(self, model_artifacts, sample_post: dict) -> None:
        pipeline, model = model_artifacts
        result = predict(sample_post, pipeline, model)
        assert result.latency_ms >= 0.0


class TestToDense:
    def test_dense_array_passthrough(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        out = _to_dense(arr)
        np.testing.assert_array_equal(out, arr)

    def test_sparse_matrix_converted(self) -> None:
        from scipy.sparse import csr_matrix
        sparse = csr_matrix([[1, 0], [0, 2]])
        out = _to_dense(sparse)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 2)


@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_train_various_sizes(n_samples: int) -> None:
    """Training should succeed for multiple dataset sizes."""
    samples, labels = generate_synthetic_data(n=n_samples)
    result = train(samples, labels, n_folds=2)
    assert isinstance(result, TrainResult)
    assert result.train_samples == n_samples
