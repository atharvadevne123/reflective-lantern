"""Tests for PSI drift metric."""

from __future__ import annotations

import pytest


class TestComputePSI:
    def test_same_distribution_psi_near_zero(self):
        from app.psi import compute_psi

        ref = list(range(50))
        result = compute_psi(ref, list(range(50)))
        assert result["psi"] < 0.1

    def test_very_different_distributions_psi_high(self):
        from app.psi import compute_psi

        ref = [1.0] * 50
        cur = [100.0] * 50
        result = compute_psi(ref, cur)
        assert result["drift_level"] == "significant"

    def test_result_has_psi_and_level(self):
        from app.psi import compute_psi

        result = compute_psi(list(range(20)), list(range(20, 40)))
        assert "psi" in result
        assert "drift_level" in result

    def test_insufficient_data(self):
        from app.psi import compute_psi

        result = compute_psi([1.0, 2.0], [3.0])
        assert result["drift_level"] == "insufficient_data"

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_psi_non_negative(self, n):
        from app.psi import compute_psi

        result = compute_psi(list(range(n)), list(range(n // 2, n + n // 2)))
        assert result["psi"] >= 0


class TestPsiReport:
    def test_returns_dict_per_feature(self):
        from app.psi import psi_report

        data = {
            "consumption": (list(range(50)), list(range(50))),
            "temperature": (list(range(50)), list(range(25, 75))),
        }
        result = psi_report(data)
        assert "consumption" in result
        assert "temperature" in result

    def test_summary_key_present(self):
        from app.psi import psi_report

        data = {"feat": (list(range(50)), list(range(50)))}
        result = psi_report(data)
        assert "summary" in result
        assert "max_psi" in result["summary"]

    def test_identical_distributions_stable(self):
        from app.psi import psi_report

        ref = list(range(100))
        result = psi_report({"x": (ref, list(ref))})
        assert result["x"]["drift_level"] == "stable"

    def test_empty_report_has_summary(self):
        from app.psi import psi_report

        result = psi_report({})
        assert isinstance(result, dict)

    def test_psi_returns_bins_used(self):
        from app.psi import compute_psi

        result = compute_psi(list(range(20)), list(range(20)))
        assert "bins_used" in result


class TestSpikeSeverity:
    def test_normal_value(self):
        from app.demand_spike import spike_severity

        assert spike_severity(5.0, 5.0, 1.0) == "normal"

    def test_moderate_spike(self):
        from app.demand_spike import spike_severity

        # z=3.0 > threshold=2.5
        assert spike_severity(8.0, 5.0, 1.0, z_threshold=2.5) == "moderate"

    def test_zero_std_returns_normal(self):
        from app.demand_spike import spike_severity

        assert spike_severity(100.0, 5.0, 0.0) == "normal"

    @pytest.mark.parametrize(
        "value,expected", [(5.5, "normal"), (8.0, "moderate"), (15.0, "high"), (30.0, "critical")]
    )
    def test_severity_levels(self, value, expected):
        from app.demand_spike import spike_severity

        result = spike_severity(value, mean=5.0, std=1.0, z_threshold=2.5)
        assert result == expected


class TestPSIEdgeCases:
    def test_constant_reference_and_current_stable(self) -> None:
        from app.psi import compute_psi

        result = compute_psi([5.0] * 50, [5.0] * 50)
        assert result["drift_level"] in ("stable", "minor")

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_custom_bins_accepted(self, bins: int) -> None:
        from app.psi import compute_psi

        result = compute_psi(list(range(50)), list(range(50)), bins=bins)
        assert result["bins_used"] <= bins

    def test_psi_report_max_psi_is_max_of_features(self) -> None:
        from app.psi import psi_report

        data = {
            "a": (list(range(50)), list(range(50))),
            "b": (list(range(50)), list(range(100, 150))),
        }
        result = psi_report(data)
        max_feature_psi = max(result["a"]["psi"], result["b"]["psi"])
        assert result["summary"]["max_psi"] == pytest.approx(max_feature_psi, rel=0.01)
