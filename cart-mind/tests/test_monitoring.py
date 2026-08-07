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


class TestComputePSI:
    def test_identical_distributions_are_stable(self):
        from app.monitoring import compute_psi

        rng = np.random.default_rng(0)
        result = compute_psi(rng.normal(0, 1, 1000).tolist(), rng.normal(0, 1, 1000).tolist())
        assert result["severity"] == "stable"
        assert result["drift_detected"] is False

    def test_shifted_distribution_flagged_major(self):
        from app.monitoring import compute_psi

        rng = np.random.default_rng(1)
        result = compute_psi(rng.normal(0, 1, 1000).tolist(), rng.normal(3, 1, 1000).tolist())
        assert result["severity"] == "major"
        assert result["drift_detected"] is True

    def test_psi_is_non_negative(self):
        from app.monitoring import compute_psi

        rng = np.random.default_rng(2)
        result = compute_psi(rng.uniform(0, 1, 500).tolist(), rng.uniform(0, 1, 500).tolist())
        assert result["psi"] >= 0.0

    def test_insufficient_data_returns_reason(self):
        from app.monitoring import compute_psi

        result = compute_psi([1.0, 2.0], [3.0])
        assert result["psi"] is None
        assert result["reason"] == "insufficient_data"

    def test_constant_reference_is_degenerate(self):
        from app.monitoring import compute_psi

        result = compute_psi([5.0] * 100, [5.0] * 100)
        assert result["psi"] is None
        assert result["reason"] == "degenerate_reference"

    def test_empty_bins_do_not_produce_infinity(self):
        """Disjoint supports must yield a finite PSI, not inf from a zero bin."""
        from app.monitoring import compute_psi

        rng = np.random.default_rng(3)
        result = compute_psi(rng.uniform(0, 1, 300).tolist(), rng.uniform(100, 101, 300).tolist())
        assert result["psi"] is not None
        assert np.isfinite(result["psi"])

    def test_psi_grows_with_shift_magnitude(self):
        from app.monitoring import compute_psi

        rng = np.random.default_rng(4)
        ref = rng.normal(0, 1, 1000).tolist()
        small = compute_psi(ref, rng.normal(0.3, 1, 1000).tolist())["psi"]
        large = compute_psi(ref, rng.normal(2.0, 1, 1000).tolist())["psi"]
        assert large > small

    def test_psi_converges_toward_zero_with_sample_size(self):
        """Unlike a KS p-value, PSI must not grow more alarming with sample size.

        For two samples from the same distribution a KS test grows more
        significant as n rises. PSI does the opposite: bin frequencies stabilise
        and the score converges toward zero. Small samples read noisy-high
        (~20 observations per bin at n=200), which is why the implementation
        docstring asks for a few hundred observations before acting on a value.
        """
        from app.monitoring import compute_psi

        rng = np.random.default_rng(5)
        small = compute_psi(rng.normal(0, 1, 200).tolist(), rng.normal(0, 1, 200).tolist())["psi"]
        large = compute_psi(rng.normal(0, 1, 5000).tolist(), rng.normal(0, 1, 5000).tolist())[
            "psi"
        ]
        assert large <= small
        assert large < 0.05

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_various_bin_counts(self, bins):
        from app.monitoring import compute_psi

        rng = np.random.default_rng(bins)
        result = compute_psi(
            rng.normal(0, 1, 500).tolist(), rng.normal(0, 1, 500).tolist(), bins=bins
        )
        assert result["psi"] is not None


class TestCheckAllFeaturesReportsPSI:
    def test_results_include_psi_fields(self, db_session):
        from app.monitoring import check_all_features, update_reference_window

        rng = np.random.default_rng(11)
        update_reference_window("psi_feat", rng.normal(0, 1, 400).tolist())
        results = check_all_features({"psi_feat": rng.normal(0, 1, 300).tolist()}, db_session)
        assert "psi" in results["psi_feat"]
        assert "psi_severity" in results["psi_feat"]

    def test_ks_and_psi_agree_on_large_shift(self, db_session):
        from app.monitoring import check_all_features, update_reference_window

        rng = np.random.default_rng(12)
        update_reference_window("shift_feat", rng.normal(0, 1, 400).tolist())
        results = check_all_features({"shift_feat": rng.normal(5, 1, 300).tolist()}, db_session)
        assert results["shift_feat"]["drift_detected"] is True
        assert results["shift_feat"]["psi_severity"] == "major"

    def test_drift_rows_persisted(self, db_session):
        from app.database import DriftLog
        from app.monitoring import check_all_features, update_reference_window

        rng = np.random.default_rng(13)
        update_reference_window("persist_feat", rng.normal(0, 1, 400).tolist())
        before = db_session.query(DriftLog).count()
        check_all_features({"persist_feat": rng.normal(0, 1, 300).tolist()}, db_session)
        assert db_session.query(DriftLog).count() > before


class TestResetReferenceWindow:
    """Tests for the reset_reference_window helper added in improvement run."""

    def test_reset_specific_feature(self):
        from app.monitoring import (
            get_reference_window,
            reset_reference_window,
            update_reference_window,
        )

        update_reference_window("feat_a", [1.0, 2.0, 3.0])
        update_reference_window("feat_b", [4.0, 5.0, 6.0])
        reset_reference_window("feat_a")
        assert get_reference_window("feat_a") == []
        assert len(get_reference_window("feat_b")) == 3

    def test_reset_all_features(self):
        from app.monitoring import (
            get_reference_window,
            reset_reference_window,
            update_reference_window,
        )

        update_reference_window("x", [1.0])
        update_reference_window("y", [2.0])
        reset_reference_window()
        assert get_reference_window("x") == []
        assert get_reference_window("y") == []

    def test_reset_absent_feature_is_noop(self):
        from app.monitoring import reset_reference_window

        reset_reference_window("does_not_exist")  # must not raise

    def test_reset_then_repopulate(self):
        from app.monitoring import (
            get_reference_window,
            reset_reference_window,
            update_reference_window,
        )

        update_reference_window("fresh", [1.0, 2.0])
        reset_reference_window("fresh")
        update_reference_window("fresh", [9.0, 8.0])
        assert get_reference_window("fresh") == [9.0, 8.0]
