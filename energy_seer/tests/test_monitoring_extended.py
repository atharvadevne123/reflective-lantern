"""Extended tests for energy_seer app/monitoring.py."""

from __future__ import annotations

import pytest

from app.monitoring import summarise_anomalies


class TestComputeDriftExtended:
    def test_insufficient_reference_data(self) -> None:
        from app.monitoring import compute_drift
        result = compute_drift([0.1, 0.2], [0.1] * 20)
        assert result.get("error") == "insufficient_data"
        assert result["drift_detected"] is False

    def test_insufficient_current_data(self) -> None:
        from app.monitoring import compute_drift
        result = compute_drift([0.1] * 20, [0.5, 0.6])
        assert result["drift_detected"] is False

    def test_identical_distributions_no_drift(self) -> None:
        from app.monitoring import compute_drift
        data = [float(i) for i in range(50)]
        result = compute_drift(data, data[:])
        assert result["drift_detected"] is False

    def test_very_different_distributions_drift_detected(self) -> None:
        from app.monitoring import compute_drift
        ref = [0.0] * 50
        cur = [100.0] * 50
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True

    def test_ks_statistic_in_range(self) -> None:
        from app.monitoring import compute_drift
        ref = [float(i) for i in range(30)]
        cur = [float(i + 5) for i in range(30)]
        result = compute_drift(ref, cur)
        assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_p_value_in_range(self) -> None:
        from app.monitoring import compute_drift
        ref = [float(i) for i in range(20)]
        cur = [float(i) for i in range(20)]
        result = compute_drift(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_result_has_required_keys(self) -> None:
        from app.monitoring import compute_drift
        ref = list(range(10))
        result = compute_drift(ref, ref[:])
        for key in ("ks_statistic", "p_value", "drift_detected"):
            assert key in result

    @pytest.mark.parametrize("n", [5, 10, 20, 50])
    def test_various_sample_sizes(self, n: int) -> None:
        from app.monitoring import compute_drift
        data = [float(i) for i in range(n)]
        result = compute_drift(data, data[:])
        assert result["drift_detected"] is False


class TestSetReferenceDistributionsExtended:
    def test_set_and_check_all_features(self) -> None:
        from app.monitoring import check_all_features, set_reference_distributions
        ref = {"temp": [float(i) for i in range(50)]}
        set_reference_distributions(ref)
        result = check_all_features({"temp": [float(i) for i in range(50)]})
        assert "temp" in result
        assert result["temp"]["drift_detected"] is False

    def test_missing_feature_returns_insufficient_data(self) -> None:
        from app.monitoring import check_all_features, set_reference_distributions
        set_reference_distributions({})
        result = check_all_features({"unknown_feat": [1.0] * 20})
        assert "unknown_feat" in result
        assert result["unknown_feat"].get("drift_detected") is False

    def test_multiple_features_checked(self) -> None:
        from app.monitoring import check_all_features, set_reference_distributions
        refs = {
            "a": [float(i) for i in range(30)],
            "b": [float(i) for i in range(30)],
        }
        set_reference_distributions(refs)
        window = {
            "a": [float(i) for i in range(30)],
            "b": [float(i) for i in range(30)],
        }
        result = check_all_features(window)
        assert "a" in result
        assert "b" in result

    def test_reference_is_copy_not_reference(self) -> None:
        from app.monitoring import _reference_distributions, set_reference_distributions
        original = {"feat": [1.0, 2.0, 3.0, 4.0, 5.0]}
        set_reference_distributions(original)
        original["feat"].append(999.0)
        assert len(_reference_distributions["feat"]) == 5


class TestGetDriftSummaryExtended:
    def test_empty_results(self) -> None:
        from app.monitoring import get_drift_summary
        result = get_drift_summary({})
        assert result["total_features"] == 0
        assert result["drifted_count"] == 0

    def test_all_stable(self) -> None:
        from app.monitoring import get_drift_summary
        feature_results = {
            "a": {"drift_detected": False, "ks_statistic": 0.1, "p_value": 0.5},
            "b": {"drift_detected": False, "ks_statistic": 0.05, "p_value": 0.8},
        }
        result = get_drift_summary(feature_results)
        assert result["drifted_count"] == 0
        assert result["drifted_features"] == []

    def test_some_drifted(self) -> None:
        from app.monitoring import get_drift_summary
        feature_results = {
            "a": {"drift_detected": True, "ks_statistic": 0.9, "p_value": 0.01},
            "b": {"drift_detected": False, "ks_statistic": 0.1, "p_value": 0.5},
        }
        result = get_drift_summary(feature_results)
        assert result["drifted_count"] == 1
        assert "a" in result["drifted_features"]

    def test_error_features_reported(self) -> None:
        from app.monitoring import get_drift_summary
        feature_results = {
            "x": {"drift_detected": False, "error": "insufficient_data"},
        }
        result = get_drift_summary(feature_results)
        assert "x" in result["error_features"]
        assert result["drifted_count"] == 0

    def test_drift_rate_calculation(self) -> None:
        from app.monitoring import get_drift_summary
        feature_results = {
            "a": {"drift_detected": True},
            "b": {"drift_detected": True},
            "c": {"drift_detected": False},
            "d": {"drift_detected": False},
        }
        result = get_drift_summary(feature_results)
        assert result["drift_rate"] == pytest.approx(0.5)

    def test_all_drifted(self) -> None:
        from app.monitoring import get_drift_summary
        feature_results = {
            "p": {"drift_detected": True},
            "q": {"drift_detected": True},
        }
        result = get_drift_summary(feature_results)
        assert result["drift_rate"] == pytest.approx(1.0)


class TestSummariseAnomaliesExtended:
    def test_empty_list_returns_zero(self) -> None:
        result = summarise_anomalies([])
        assert result["total"] == 0
        assert result["anomaly_count"] == 0
        assert result["anomaly_rate"] == pytest.approx(0.0)

    def test_all_normal_zero_anomalies(self) -> None:
        results = [{"is_anomaly": False, "severity": "none", "anomaly_type": "none"}] * 5
        result = summarise_anomalies(results)
        assert result["anomaly_count"] == 0

    def test_all_anomalies_full_rate(self) -> None:
        results = [{"is_anomaly": True, "severity": "high", "anomaly_type": "spike"}] * 4
        result = summarise_anomalies(results)
        assert result["anomaly_count"] == 4
        assert result["anomaly_rate"] == pytest.approx(1.0)

    def test_mixed_severities_counted(self) -> None:
        results = [
            {"is_anomaly": True, "severity": "high", "anomaly_type": "spike"},
            {"is_anomaly": True, "severity": "critical", "anomaly_type": "drop"},
            {"is_anomaly": False, "severity": "none", "anomaly_type": "none"},
        ]
        result = summarise_anomalies(results)
        assert result["severity_counts"]["high"] == 1
        assert result["severity_counts"]["critical"] == 1
        assert result["severity_counts"]["none"] == 1

    def test_type_counts_in_result(self) -> None:
        results = [
            {"is_anomaly": True, "severity": "high", "anomaly_type": "spike"},
            {"is_anomaly": True, "severity": "low", "anomaly_type": "spike"},
        ]
        result = summarise_anomalies(results)
        assert result["type_counts"]["spike"] == 2

    def test_anomaly_rate_between_zero_and_one(self) -> None:
        results = [
            {"is_anomaly": True, "severity": "high", "anomaly_type": "spike"},
            {"is_anomaly": False, "severity": "none", "anomaly_type": "none"},
        ]
        result = summarise_anomalies(results)
        assert 0.0 <= result["anomaly_rate"] <= 1.0
