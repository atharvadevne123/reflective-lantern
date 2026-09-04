"""Tests for app.cost_estimator module."""

from __future__ import annotations

import pytest

from app.cost_estimator import (
    ResourceSpec,
    compare_specs,
    estimate_cost,
    monthly_estimate,
)


class TestResourceSpec:
    def test_valid_spec(self):
        s = ResourceSpec(cpu_cores=4, memory_gb=16)
        assert s.gpu_count == 0

    def test_invalid_cpu_raises(self):
        with pytest.raises(ValueError, match="cpu_cores"):
            ResourceSpec(cpu_cores=0, memory_gb=8)

    def test_invalid_memory_raises(self):
        with pytest.raises(ValueError, match="memory_gb"):
            ResourceSpec(cpu_cores=2, memory_gb=-1)

    def test_invalid_gpu_raises(self):
        with pytest.raises(ValueError, match="gpu_count"):
            ResourceSpec(cpu_cores=2, memory_gb=8, gpu_count=-1)

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError, match="duration_hours"):
            ResourceSpec(cpu_cores=2, memory_gb=8, duration_hours=0)


class TestEstimateCost:
    def test_cpu_only_cost(self):
        spec = ResourceSpec(cpu_cores=4, memory_gb=16, duration_hours=1)
        bd = estimate_cost(spec, cpu_rate=1.0, memory_rate=0.0, gpu_rate=0.0)
        assert bd.cpu_cost_usd == pytest.approx(4.0)
        assert bd.gpu_cost_usd == pytest.approx(0.0)

    def test_total_is_sum(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=4, gpu_count=1, duration_hours=2)
        bd = estimate_cost(spec, cpu_rate=1.0, memory_rate=1.0, gpu_rate=1.0)
        assert bd.total_usd == pytest.approx(bd.cpu_cost_usd + bd.memory_cost_usd + bd.gpu_cost_usd)

    def test_to_dict_keys(self):
        spec = ResourceSpec(cpu_cores=1, memory_gb=1)
        d = estimate_cost(spec).to_dict()
        assert set(d.keys()) == {"cpu_cost_usd", "memory_cost_usd", "gpu_cost_usd", "total_usd"}

    def test_gpu_cost_positive(self):
        spec = ResourceSpec(cpu_cores=1, memory_gb=4, gpu_count=2, duration_hours=1)
        bd = estimate_cost(spec, gpu_rate=3.0)
        assert bd.gpu_cost_usd == pytest.approx(6.0)

    @pytest.mark.parametrize("duration", [0.5, 1.0, 8.0, 24.0])
    def test_cost_scales_with_duration(self, duration):
        base = estimate_cost(ResourceSpec(cpu_cores=1, memory_gb=1, duration_hours=1), cpu_rate=1.0, memory_rate=1.0)
        scaled = estimate_cost(
            ResourceSpec(cpu_cores=1, memory_gb=1, duration_hours=duration), cpu_rate=1.0, memory_rate=1.0
        )
        assert scaled.total_usd == pytest.approx(base.total_usd * duration, rel=1e-6)


class TestMonthlyEstimate:
    def test_monthly_higher_than_hourly(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=8, duration_hours=1)
        hourly = estimate_cost(spec)
        monthly = monthly_estimate(spec)
        assert monthly.total_usd > hourly.total_usd

    def test_zero_hours_per_day_invalid(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=8)
        with pytest.raises(ValueError):
            monthly_estimate(spec, hours_per_day=0)


class TestCompareSpecs:
    def test_returns_sorted_by_cost(self):
        specs = [
            ResourceSpec(cpu_cores=8, memory_gb=32),
            ResourceSpec(cpu_cores=1, memory_gb=4),
        ]
        results = compare_specs(specs)
        assert results[0]["total_usd"] <= results[1]["total_usd"]

    def test_labels_applied(self):
        specs = [ResourceSpec(cpu_cores=1, memory_gb=4)]
        results = compare_specs(specs, labels=["tiny"])
        assert results[0]["label"] == "tiny"

    def test_default_labels_generated(self):
        specs = [ResourceSpec(cpu_cores=1, memory_gb=4), ResourceSpec(cpu_cores=2, memory_gb=8)]
        results = compare_specs(specs)
        labels = {r["label"] for r in results}
        assert "spec-0" in labels or "spec-1" in labels


class TestResourceSpecDefaults:
    def test_gpu_count_defaults_to_zero(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=8)
        assert spec.gpu_count == 0

    def test_duration_defaults_to_one_hour(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=8)
        assert spec.duration_hours == 1.0

    @pytest.mark.parametrize("cpu,mem", [(1, 1), (8, 64), (64, 256)])
    def test_valid_spec_combinations(self, cpu: int, mem: int) -> None:
        spec = ResourceSpec(cpu_cores=cpu, memory_gb=mem)
        assert spec.cpu_cores == cpu
        assert spec.memory_gb == mem


class TestEstimateCostEdgeCases:
    def test_all_zero_rates_gives_zero_total(self):
        spec = ResourceSpec(cpu_cores=4, memory_gb=16, gpu_count=2, duration_hours=8)
        bd = estimate_cost(spec, cpu_rate=0.0, memory_rate=0.0, gpu_rate=0.0)
        assert bd.total_usd == pytest.approx(0.0)

    def test_memory_cost_scales_with_gb(self):
        spec1 = ResourceSpec(cpu_cores=1, memory_gb=4, duration_hours=1)
        spec2 = ResourceSpec(cpu_cores=1, memory_gb=8, duration_hours=1)
        bd1 = estimate_cost(spec1, cpu_rate=0.0, memory_rate=1.0, gpu_rate=0.0)
        bd2 = estimate_cost(spec2, cpu_rate=0.0, memory_rate=1.0, gpu_rate=0.0)
        assert bd2.memory_cost_usd == pytest.approx(bd1.memory_cost_usd * 2)

    def test_to_dict_all_values_non_negative(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=8, gpu_count=1, duration_hours=4)
        d = estimate_cost(spec).to_dict()
        assert all(v >= 0 for v in d.values())

    def test_fractional_duration_accepted(self):
        spec = ResourceSpec(cpu_cores=2, memory_gb=4, duration_hours=0.25)
        bd = estimate_cost(spec, cpu_rate=1.0, memory_rate=0.0, gpu_rate=0.0)
        assert bd.cpu_cost_usd == pytest.approx(0.5)  # 2 cores * 0.25h
