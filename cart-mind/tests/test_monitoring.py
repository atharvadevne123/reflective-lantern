"""Tests for drift detection and prediction logging."""

from __future__ import annotations

import numpy as np
import pytest


class TestComputeDrift:
    def test_no_drift_same_distribution(self):
        from app.monitoring import compute_drift

        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 200).tolist()
        cur = rng.normal(0, 1, 200).tolist()
        result = compute_drift(ref, cur)
        assert "ks_statistic" in result
        assert "drift_detected" in result

    def test_drift_detected_different_distribution(self):
        from app.monitoring import compute_drift

        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 300).tolist()
        cur = rng.normal(10, 1, 300).tolist()  # clearly shifted
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_insufficient_data_returns_reason(self):
        from app.monitoring import compute_drift

        result = compute_drift([1.0, 2.0], [3.0, 4.0])
        assert result["drift_detected"] is False
        assert result.get("reason") == "insufficient_data"

    def test_ks_statistic_in_range(self):
        from app.monitoring import compute_drift

        rng = np.random.default_rng(1)
        ref = rng.uniform(0, 1, 100).tolist()
        cur = rng.uniform(0, 1, 100).tolist()
        result = compute_drift(ref, cur)
        if result["ks_statistic"] is not None:
            assert 0.0 <= result["ks_statistic"] <= 1.0

    @pytest.mark.parametrize("n", [20, 50, 100, 500])
    def test_drift_with_various_sample_sizes(self, n):
        from app.monitoring import compute_drift

        rng = np.random.default_rng(n)
        ref = rng.normal(0, 1, n).tolist()
        cur = rng.normal(0, 1, n).tolist()
        result = compute_drift(ref, cur)
        assert isinstance(result["drift_detected"], bool)


class TestReferenceWindow:
    def test_update_and_retrieve(self):
        from app.monitoring import get_reference_window, update_reference_window

        update_reference_window("test_feat", [1.0, 2.0, 3.0])
        window = get_reference_window("test_feat")
        assert len(window) >= 3

    def test_window_capped_at_max_size(self):
        from app.monitoring import (
            REFERENCE_WINDOW_SIZE,
            get_reference_window,
            update_reference_window,
        )

        feature = "cap_test"
        big_batch = list(range(REFERENCE_WINDOW_SIZE + 100))
        update_reference_window(feature, big_batch)
        window = get_reference_window(feature)
        assert len(window) <= REFERENCE_WINDOW_SIZE


class TestLogPrediction:
    def test_log_persists_record(self, db_session):
        from app.database import PredictionLog
        from app.monitoring import log_prediction

        log_prediction(
            db=db_session,
            correlation_id="cid-test",
            user_id="u_test",
            item_id="i_test",
            prediction_type="intent",
            score=0.75,
            latency_ms=12.3,
        )
        row = db_session.query(PredictionLog).filter_by(correlation_id="cid-test").first()
        assert row is not None
        assert abs(row.score - 0.75) < 1e-4

    def test_log_without_item_id(self, db_session):
        from app.database import PredictionLog
        from app.monitoring import log_prediction

        log_prediction(
            db=db_session,
            correlation_id="cid-noid",
            user_id="u_test2",
            item_id=None,
            prediction_type="recommend",
            score=0.55,
            latency_ms=8.0,
        )
        row = db_session.query(PredictionLog).filter_by(correlation_id="cid-noid").first()
        assert row is not None
        assert row.item_id is None


class TestCounters:
    def test_record_increments_total(self):
        from app.monitoring import _Counters

        c = _Counters()
        c.record("intent", 10.0)
        assert c.total_requests == 1

    def test_record_increments_route(self):
        from app.monitoring import _Counters

        c = _Counters()
        c.record("recommend", 5.0)
        assert c.recommend_requests == 1

    def test_snapshot_returns_dict(self):
        from app.monitoring import _Counters

        c = _Counters()
        c.record("similar", 20.0)
        snap = c.snapshot()
        assert isinstance(snap, dict)
        assert "p50_latency_ms" in snap
