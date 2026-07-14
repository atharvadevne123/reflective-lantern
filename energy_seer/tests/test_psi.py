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
