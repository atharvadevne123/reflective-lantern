"""Extended tests for energy_seer app/psi.py PSI drift detection."""

from __future__ import annotations

import pytest

from app.psi import (
    PSI_MODERATE_THRESHOLD,
    PSI_STABLE_THRESHOLD,
    compute_psi,
    psi_alert,
    psi_batch_alert,
    psi_feature_rank,
    psi_report,
    psi_series_trend,
)


class TestComputePsi:
    def test_insufficient_data_short_ref(self) -> None:
        result = compute_psi([0.1, 0.2, 0.3], [0.1] * 10)
        assert result["drift_level"] == "insufficient_data"

    def test_insufficient_data_short_cur(self) -> None:
        result = compute_psi([0.1] * 10, [0.2, 0.3])
        assert result["drift_level"] == "insufficient_data"

    def test_identical_distributions_stable(self) -> None:
        data = [float(i) for i in range(20)]
        result = compute_psi(data, data)
        assert result["drift_level"] == "stable"
        assert result["psi"] == pytest.approx(0.0, abs=0.01)

    def test_very_different_distributions_significant(self) -> None:
        ref = [0.0] * 20
        cur = [1.0] * 20
        result = compute_psi(ref, cur)
        assert result["drift_level"] == "significant"

    def test_returns_dict_with_required_keys(self) -> None:
        result = compute_psi(list(range(10)), list(range(10, 20)))
        assert "psi" in result
        assert "drift_level" in result
        assert "bins_used" in result

    def test_psi_non_negative(self) -> None:
        result = compute_psi(list(range(20)), [x + 5 for x in range(20)])
        assert float(result["psi"]) >= 0.0

    def test_constant_distribution_stable(self) -> None:
        result = compute_psi([5.0] * 10, [5.0] * 10)
        assert result["drift_level"] == "stable"

    def test_bins_used_matches_param(self) -> None:
        result = compute_psi(list(range(20)), list(range(20)), bins=5)
        assert result["bins_used"] == 5


class TestPsiThresholds:
    def test_stable_lt_moderate(self) -> None:
        assert PSI_STABLE_THRESHOLD < PSI_MODERATE_THRESHOLD

    def test_stable_threshold_value(self) -> None:
        assert pytest.approx(0.1) == PSI_STABLE_THRESHOLD

    def test_moderate_threshold_value(self) -> None:
        assert pytest.approx(0.25) == PSI_MODERATE_THRESHOLD


class TestPsiAlert:
    def test_stable_no_alert(self) -> None:
        result = psi_alert(0.05)
        assert result["alert"] is False
        assert result["drift_level"] == "stable"

    def test_moderate_alert(self) -> None:
        result = psi_alert(0.15)
        assert result["alert"] is True
        assert result["drift_level"] == "moderate"

    def test_significant_alert(self) -> None:
        result = psi_alert(0.30)
        assert result["alert"] is True
        assert result["drift_level"] == "significant"

    def test_message_contains_feature_name(self) -> None:
        result = psi_alert(0.30, feature_name="price")
        assert "price" in result["message"]


class TestPsiReport:
    def test_returns_dict(self) -> None:
        fd = {"feat_a": (list(range(10)), list(range(10, 20)))}
        result = psi_report(fd)
        assert isinstance(result, dict)

    def test_has_summary_key(self) -> None:
        fd = {"feat_a": (list(range(10)), list(range(10, 20)))}
        result = psi_report(fd)
        assert "summary" in result

    def test_summary_has_max_psi(self) -> None:
        fd = {"feat_a": (list(range(10)), list(range(10, 20)))}
        result = psi_report(fd)
        assert "max_psi" in result["summary"]


class TestPsiSeriesTrend:
    def test_empty_returns_stable(self) -> None:
        result = psi_series_trend([])
        assert result["drift_level"] == "stable"
        assert result["escalating"] is False

    def test_escalating_when_last_exceeds_mean(self) -> None:
        result = psi_series_trend([0.05, 0.05, 0.05, 0.30])
        assert result["escalating"] is True

    def test_not_escalating_when_flat(self) -> None:
        result = psi_series_trend([0.05, 0.05, 0.05, 0.06])
        assert result["escalating"] is False


class TestPsiFeatureRank:
    def test_returns_list(self) -> None:
        fd = {
            "a": (list(range(10)), list(range(5, 15))),
            "b": (list(range(10)), list(range(10, 20))),
        }
        result = psi_feature_rank(fd)
        assert isinstance(result, list)

    def test_sorted_by_psi_descending(self) -> None:
        fd = {
            "a": ([0.0] * 10, [0.0] * 10),  # zero PSI
            "b": ([0.0] * 10, [1.0] * 10),  # high PSI
        }
        result = psi_feature_rank(fd)
        psi_values = [float(r["psi"]) for r in result]
        assert psi_values == sorted(psi_values, reverse=True)


class TestPsiReportExtended:
    def test_empty_feature_dict_returns_empty(self) -> None:
        result = psi_report({})
        assert result == {}

    def test_multiple_features_all_in_result(self) -> None:
        fd = {
            "temp": (list(range(10)), list(range(10, 20))),
            "humidity": (list(range(10)), list(range(10, 20))),
        }
        result = psi_report(fd)
        assert "temp" in result
        assert "humidity" in result
        assert "summary" in result

    def test_summary_max_psi_is_non_negative(self) -> None:
        fd = {"a": (list(range(10)), list(range(10)))}
        result = psi_report(fd)
        assert float(result["summary"]["max_psi"]) >= 0.0

    def test_summary_worst_feature_identified(self) -> None:
        fd = {
            "stable": ([0.0] * 10, [0.0] * 10),
            "drifted": ([0.0] * 10, [1.0] * 10),
        }
        result = psi_report(fd)
        assert result["summary"]["worst_feature"] == "drifted"

    def test_drifted_features_counted(self) -> None:
        fd = {
            "a": ([0.0] * 10, [1.0] * 10),
            "b": ([0.0] * 10, [0.0] * 10),
        }
        result = psi_report(fd)
        assert int(result["summary"]["drifted_features"]) >= 1


class TestPsiSeriesTrendExtended:
    def test_single_value_not_escalating(self) -> None:
        result = psi_series_trend([0.05])
        assert result["escalating"] is False

    def test_mean_psi_correct(self) -> None:
        result = psi_series_trend([0.1, 0.2, 0.3])
        assert result["mean_psi"] == pytest.approx(0.2, abs=0.001)

    def test_max_psi_correct(self) -> None:
        result = psi_series_trend([0.05, 0.3, 0.15])
        assert result["max_psi"] == pytest.approx(0.3, abs=0.001)

    def test_latest_drift_level_reported(self) -> None:
        result = psi_series_trend([0.3])
        assert result["drift_level"] == "significant"

    @pytest.mark.parametrize("psi_val,expected_level", [
        (0.05, "stable"),
        (0.15, "moderate"),
        (0.30, "significant"),
    ])
    def test_drift_levels_at_boundaries(self, psi_val: float, expected_level: str) -> None:
        result = psi_series_trend([psi_val])
        assert result["drift_level"] == expected_level


class TestPsiBatchAlert:
    def test_empty_dict_returns_empty(self) -> None:
        result = psi_batch_alert({})
        assert result == []

    def test_one_feature_one_alert(self) -> None:
        result = psi_batch_alert({"temp": 0.3})
        assert len(result) == 1
        assert result[0]["alert"] is True

    def test_multiple_features_all_returned(self) -> None:
        result = psi_batch_alert({"a": 0.05, "b": 0.15, "c": 0.30})
        assert len(result) == 3

    def test_stable_feature_no_alert(self) -> None:
        result = psi_batch_alert({"x": 0.05})
        assert result[0]["alert"] is False

    def test_message_contains_feature_name(self) -> None:
        result = psi_batch_alert({"humidity": 0.30})
        assert "humidity" in result[0]["message"]

    def test_drift_level_assigned(self) -> None:
        result = psi_batch_alert({"f": 0.15})
        assert result[0]["drift_level"] == "moderate"


class TestPsiSeriesTrendExtended2:
    def test_empty_returns_stable(self) -> None:
        result = psi_series_trend([])
        assert result["escalating"] is False
        assert result["drift_level"] == "stable"

    def test_single_value_not_escalating(self) -> None:
        result = psi_series_trend([0.05])
        assert result["escalating"] is False

    def test_escalating_when_last_exceeds_mean(self) -> None:
        vals = [0.01, 0.01, 0.01, 0.50]
        result = psi_series_trend(vals)
        assert result["escalating"] is True

    def test_max_psi_correct(self) -> None:
        vals = [0.1, 0.3, 0.2]
        result = psi_series_trend(vals)
        assert result["max_psi"] == pytest.approx(0.3)


class TestPsiFeatureRankExtended:
    def test_empty_dict_returns_empty(self) -> None:
        assert psi_feature_rank({}) == []

    def test_sorted_descending(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        ref_a = list(rng.normal(0, 1, 200))
        cur_a = list(rng.normal(5, 1, 200))
        ref_b = list(rng.normal(0, 1, 200))
        cur_b = list(rng.normal(0.1, 1, 200))
        result = psi_feature_rank({"a": (ref_a, cur_a), "b": (ref_b, cur_b)})
        assert len(result) == 2
        assert result[0]["psi"] >= result[1]["psi"]
