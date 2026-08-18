"""Tests for FAISS/sklearn anomaly root cause analysis."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture()
def built_index():
    """Build a small anomaly index."""
    from app import anomaly

    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (50, 8)).astype(np.float32)
    labels = [{"event_id": i, "cause": f"cause-{i % 3}"} for i in range(50)]
    feature_names = [f"feat_{i}" for i in range(8)]
    anomaly.build_index(X, labels, feature_names)
    yield anomaly
    anomaly._INDEX = None
    anomaly._NN_MODEL = None
    anomaly._INDEX_LABELS = []


class TestBuildIndex:
    def test_index_is_queryable(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=3)
        assert len(results) == 3

    def test_results_have_distance(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=2)
        for r in results:
            assert "distance" in r
            assert r["distance"] >= 0.0

    def test_results_preserve_labels(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        results = built_index.find_similar_anomalies(query, k=1)
        assert "cause" in results[0]

    def test_empty_index_returns_empty(self):
        from app import anomaly

        anomaly._INDEX = None
        anomaly._NN_MODEL = None
        result = anomaly.find_similar_anomalies(np.zeros(8, dtype=np.float32))
        assert result == []


class TestExplainAnomaly:
    def test_explain_returns_contributors(self, built_index):
        query = np.array([0, 0, 0, 10.0, 0, 0, 0, 0], dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        assert "top_contributors" in explanation

    def test_extreme_feature_ranked_first(self, built_index):
        query = np.array([0, 0, 0, 100.0, 0, 0, 0, 0], dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        top = explanation["top_contributors"][0]
        assert top["feature"] == "feat_3"

    def test_explain_includes_similar_anomalies(self, built_index):
        query = np.zeros(8, dtype=np.float32)
        explanation = built_index.explain_anomaly(query)
        assert "similar_historical_anomalies" in explanation

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_top_k_respected(self, built_index, top_k):
        query = np.arange(8, dtype=np.float32)
        explanation = built_index.explain_anomaly(query, top_k=top_k)
        assert len(explanation["top_contributors"]) == top_k


class TestAnomalyScoreSummary:
    def test_empty_returns_zeros(self):
        from app.anomaly import anomaly_score_summary

        result = anomaly_score_summary([])
        assert result == {"mean": 0.0, "max": 0.0, "min": 0.0, "p90": 0.0}

    def test_single_value(self):
        from app.anomaly import anomaly_score_summary

        result = anomaly_score_summary([5.0])
        assert result["mean"] == pytest.approx(5.0)
        assert result["max"] == pytest.approx(5.0)
        assert result["min"] == pytest.approx(5.0)

    def test_p90_ordering(self):
        from app.anomaly import anomaly_score_summary

        result = anomaly_score_summary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert result["p90"] >= result["mean"]

    @pytest.mark.parametrize(
        "scores,expected_max",
        [
            ([1.0, 2.0, 3.0], 3.0),
            ([10.0, 20.0, 30.0], 30.0),
        ],
    )
    def test_parametrized_max(self, scores, expected_max):
        from app.anomaly import anomaly_score_summary

        assert anomaly_score_summary(scores)["max"] == pytest.approx(expected_max)


class TestFlagRecurringAnomalies:
    def test_empty_returns_empty(self):
        from app.anomaly import flag_recurring_anomalies

        assert flag_recurring_anomalies([]) == []

    def test_all_unique_returns_empty(self):
        from app.anomaly import flag_recurring_anomalies

        assert flag_recurring_anomalies(["a", "b", "c"]) == []

    def test_recurring_detected(self):
        from app.anomaly import flag_recurring_anomalies

        result = flag_recurring_anomalies(["a", "a", "b"])
        assert "a" in result
        assert "b" not in result

    def test_min_occurrences_respected(self):
        from app.anomaly import flag_recurring_anomalies

        result = flag_recurring_anomalies(["a", "a", "a", "b", "b"], min_occurrences=3)
        assert result == ["a"]

    def test_result_is_sorted(self):
        from app.anomaly import flag_recurring_anomalies

        result = flag_recurring_anomalies(["z", "z", "a", "a", "m", "m"])
        assert result == sorted(result)


class TestNormaliseScores:
    def test_empty_returns_empty(self):
        from app.anomaly import normalise_scores

        assert normalise_scores([]) == []

    def test_constant_returns_zeros(self):
        from app.anomaly import normalise_scores

        assert normalise_scores([5.0] * 5) == [0.0] * 5

    def test_min_is_zero_max_is_one(self):
        from app.anomaly import normalise_scores

        result = normalise_scores([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "scores,expected",
        [
            ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0]),
            ([0.0, 1.0], [0.0, 1.0]),
        ],
    )
    def test_parametrized_normalise(self, scores, expected):
        from app.anomaly import normalise_scores

        result = normalise_scores(scores)
        for r, e in zip(result, expected):
            assert r == pytest.approx(e)


@pytest.mark.parametrize("n_scores", [0, 1, 5, 10])
def test_anomaly_score_summary_various_lengths(n_scores: int) -> None:
    from app.anomaly import anomaly_score_summary

    if n_scores == 0:
        result = anomaly_score_summary([])
        assert result == {}
    else:
        scores = [float(i) for i in range(n_scores)]
        result = anomaly_score_summary(scores)
        assert "mean" in result
        assert "max" in result


@pytest.mark.parametrize(
    "scores,expected_min,expected_max",
    [
        ([0.0, 1.0], 0.0, 1.0),
        ([5.0, 5.0, 5.0], 0.0, 0.0),
        ([0.0, 0.5, 1.0], 0.0, 1.0),
    ],
)
def test_normalise_scores_range(scores: list[float], expected_min: float, expected_max: float) -> None:
    from app.anomaly import normalise_scores

    result = normalise_scores(scores)
    if result:
        assert min(result) == pytest.approx(expected_min, abs=1e-6)
        assert max(result) == pytest.approx(expected_max, abs=1e-6)


class TestAnomalyBurstCount:
    def test_single_burst_detected(self) -> None:
        from app.anomaly import anomaly_burst_count

        scores = [0.1, 0.9, 0.9, 0.9, 0.1]
        assert anomaly_burst_count(scores, threshold=0.5, min_burst=3) == 1

    def test_burst_too_short_not_counted(self) -> None:
        from app.anomaly import anomaly_burst_count

        scores = [0.9, 0.9, 0.1, 0.1]
        assert anomaly_burst_count(scores, threshold=0.5, min_burst=3) == 0

    def test_empty_series(self) -> None:
        from app.anomaly import anomaly_burst_count

        assert anomaly_burst_count([], threshold=0.5) == 0

    @pytest.mark.parametrize("min_burst,expected", [(1, 1), (2, 1), (4, 0)])
    def test_min_burst_boundary(self, min_burst: int, expected: int) -> None:
        from app.anomaly import anomaly_burst_count

        scores = [0.1, 0.9, 0.9, 0.9, 0.1]
        assert anomaly_burst_count(scores, threshold=0.5, min_burst=min_burst) == expected


class TestScorePercentileThreshold:
    def test_empty_returns_zero(self) -> None:
        from app.anomaly import score_percentile_threshold

        assert score_percentile_threshold([]) == 0.0

    def test_95th_percentile_of_range(self) -> None:
        from app.anomaly import score_percentile_threshold

        scores = list(range(1, 101))
        result = score_percentile_threshold(scores, percentile=95.0)
        assert result == pytest.approx(95.05, abs=0.5)

    def test_100th_equals_max(self) -> None:
        from app.anomaly import score_percentile_threshold

        scores = [1.0, 2.0, 3.0, 10.0]
        assert score_percentile_threshold(scores, percentile=100.0) == pytest.approx(10.0)


class TestConsecutiveAnomalyLengths:
    def test_no_anomalies_empty(self) -> None:
        from app.anomaly import consecutive_anomaly_lengths

        assert consecutive_anomaly_lengths([False, False, False]) == []

    def test_single_run_detected(self) -> None:
        from app.anomaly import consecutive_anomaly_lengths

        result = consecutive_anomaly_lengths([False, True, True, False])
        assert result == [2]

    def test_multiple_runs(self) -> None:
        from app.anomaly import consecutive_anomaly_lengths

        result = consecutive_anomaly_lengths([True, False, True, True, False])
        assert result == [1, 2]

    def test_all_anomalous(self) -> None:
        from app.anomaly import consecutive_anomaly_lengths

        result = consecutive_anomaly_lengths([True, True, True])
        assert result == [3]

    def test_empty_input(self) -> None:
        from app.anomaly import consecutive_anomaly_lengths

        assert consecutive_anomaly_lengths([]) == []


class TestAnomalyDensity:
    def test_same_length_output(self) -> None:
        from app.anomaly import anomaly_density

        scores = [0.0, 5.0, 5.0, 0.0]
        result = anomaly_density(scores, threshold=3.0)
        assert len(result) == 4

    def test_no_anomalies_all_zero(self) -> None:
        from app.anomaly import anomaly_density

        result = anomaly_density([0.0, 1.0, 0.0], threshold=5.0)
        assert all(d == 0.0 for d in result)

    def test_all_anomalies_all_one(self) -> None:
        from app.anomaly import anomaly_density

        result = anomaly_density([10.0, 10.0, 10.0], threshold=5.0, window=1)
        assert all(d == 1.0 for d in result)
