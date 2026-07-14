"""Tests for drift detection, anomaly detection, and logging."""

from __future__ import annotations

import pytest


class TestComputeDrift:
    def test_no_drift_same_distribution(self):
        from app.monitoring import compute_drift

        ref = [3.0 + i * 0.1 for i in range(50)]
        cur = [3.0 + i * 0.1 for i in range(50)]
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is False
        assert result["ks_statistic"] < 0.1

    def test_drift_detected_different_distribution(self):
        from app.monitoring import compute_drift

        ref = [1.0] * 50
        cur = [100.0] * 50
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True
        assert result["ks_statistic"] > 0.5

    def test_insufficient_data_returns_no_drift(self):
        from app.monitoring import compute_drift

        result = compute_drift([1.0, 2.0], [3.0, 4.0])
        assert result["drift_detected"] is False
        assert "error" in result

    def test_result_has_required_keys(self):
        from app.monitoring import compute_drift

        result = compute_drift(list(range(20)), list(range(20, 40)))
        assert "ks_statistic" in result
        assert "p_value" in result
        assert "drift_detected" in result

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_drift_various_sample_sizes(self, n):
        from app.monitoring import compute_drift

        ref = list(range(n))
        cur = [x + 50 for x in range(n)]
        result = compute_drift(ref, cur)
        assert isinstance(result["drift_detected"], bool)


class TestCheckAllFeatures:
    def test_returns_dict_per_feature(self):
        from app.monitoring import check_all_features, set_reference_distributions

        set_reference_distributions({"consumption_kwh": list(range(50))})
        results = check_all_features({"consumption_kwh": list(range(50, 100))})
        assert "consumption_kwh" in results

    def test_handles_missing_reference(self):
        from app.monitoring import check_all_features

        results = check_all_features({"unknown_feature": [1.0, 2.0, 3.0]})
        assert "unknown_feature" in results
        assert results["unknown_feature"]["drift_detected"] is False


class TestDetectAnomaly:
    def test_normal_consumption_not_anomaly(self):
        from app.monitoring import (
            detect_anomaly,
            set_reference_distributions,
            train_anomaly_detector,
        )

        vals = [3.0 + i * 0.1 for i in range(200)]
        set_reference_distributions({"consumption_kwh": vals})
        train_anomaly_detector(vals)
        result = detect_anomaly(3.5)
        assert "is_anomaly" in result
        assert "severity" in result

    def test_extreme_spike_flagged(self):
        from app.monitoring import (
            detect_anomaly,
            set_reference_distributions,
            train_anomaly_detector,
        )

        vals = [3.0] * 200
        set_reference_distributions({"consumption_kwh": vals})
        train_anomaly_detector(vals)
        result = detect_anomaly(999.9)
        assert result["is_anomaly"] is True

    def test_anomaly_type_spike_for_high_value(self):
        from app.monitoring import (
            detect_anomaly,
            set_reference_distributions,
            train_anomaly_detector,
        )

        vals = [3.0] * 200
        set_reference_distributions({"consumption_kwh": vals})
        train_anomaly_detector(vals)
        result = detect_anomaly(500.0)
        assert result["anomaly_type"] == "spike"

    def test_severity_levels_valid(self):
        from app.monitoring import detect_anomaly

        result = detect_anomaly(4.0)
        assert result["severity"] in {"none", "low", "medium", "high", "critical"}

    def test_result_has_z_score(self):
        from app.monitoring import detect_anomaly

        result = detect_anomaly(3.0)
        assert "z_score" in result
        assert result["z_score"] >= 0


class TestTrainAnomalyDetector:
    def test_returns_isolation_forest(self):
        from sklearn.ensemble import IsolationForest

        from app.monitoring import train_anomaly_detector

        detector = train_anomaly_detector(list(range(100)))
        assert isinstance(detector, IsolationForest)


class TestComputePredictionStats:
    def test_returns_count_zero_when_empty(self, setup_test_db):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.monitoring import compute_prediction_stats

        engine = create_engine("sqlite:///:memory:")
        from app.database import Base

        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        stats = compute_prediction_stats(db)
        assert stats["count"] == 0
        db.close()
