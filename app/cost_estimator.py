"""ML inference and training cost estimation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_USD_PER_GB_MEMORY_HOUR = 0.0125
_USD_PER_CPU_HOUR = 0.048
_USD_PER_GPU_HOUR = 2.50


@dataclass
class ResourceSpec:
    """Compute resource specification for a workload.

    Attributes:
        cpu_cores: Number of virtual CPU cores.
        memory_gb: Memory in gigabytes.
        gpu_count: Number of GPU accelerators.
        duration_hours: Wall-clock duration in hours.
    """

    cpu_cores: float
    memory_gb: float
    gpu_count: int = 0
    duration_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be >= 0")
        if self.duration_hours <= 0:
            raise ValueError("duration_hours must be positive")


@dataclass
class CostBreakdown:
    """Itemised cost estimate.

    Attributes:
        cpu_cost_usd: Cost attributed to CPU usage.
        memory_cost_usd: Cost attributed to memory usage.
        gpu_cost_usd: Cost attributed to GPU usage.
        total_usd: Sum of all cost components.
    """

    cpu_cost_usd: float
    memory_cost_usd: float
    gpu_cost_usd: float

    @property
    def total_usd(self) -> float:
        return self.cpu_cost_usd + self.memory_cost_usd + self.gpu_cost_usd

    def to_dict(self) -> dict[str, float]:
        return {
            "cpu_cost_usd": round(self.cpu_cost_usd, 6),
            "memory_cost_usd": round(self.memory_cost_usd, 6),
            "gpu_cost_usd": round(self.gpu_cost_usd, 6),
            "total_usd": round(self.total_usd, 6),
        }


def estimate_cost(
    spec: ResourceSpec,
    cpu_rate: float = _USD_PER_CPU_HOUR,
    memory_rate: float = _USD_PER_GB_MEMORY_HOUR,
    gpu_rate: float = _USD_PER_GPU_HOUR,
) -> CostBreakdown:
    """Estimate the cost for a resource specification.

    Args:
        spec: Resource specification.
        cpu_rate: Cost per CPU core per hour in USD.
        memory_rate: Cost per GB of memory per hour in USD.
        gpu_rate: Cost per GPU per hour in USD.

    Returns:
        Itemised :class:`CostBreakdown`.
    """
    cpu_cost = spec.cpu_cores * cpu_rate * spec.duration_hours
    mem_cost = spec.memory_gb * memory_rate * spec.duration_hours
    gpu_cost = spec.gpu_count * gpu_rate * spec.duration_hours
    breakdown = CostBreakdown(
        cpu_cost_usd=cpu_cost,
        memory_cost_usd=mem_cost,
        gpu_cost_usd=gpu_cost,
    )
    logger.debug(
        "Cost estimate: CPU=%.4f MEM=%.4f GPU=%.4f total=%.4f USD",
        cpu_cost,
        mem_cost,
        gpu_cost,
        breakdown.total_usd,
    )
    return breakdown


def monthly_estimate(
    spec: ResourceSpec,
    hours_per_day: float = 24.0,
    days_per_month: float = 30.0,
    **rate_kwargs,
) -> CostBreakdown:
    """Project daily resource usage to a monthly cost estimate.

    Args:
        spec: Resource spec for a single run duration.
        hours_per_day: How many hours per day the workload runs.
        days_per_month: Days in the billing period.
        **rate_kwargs: Forwarded to :func:`estimate_cost`.

    Returns:
        :class:`CostBreakdown` for the full month.
    """
    monthly_hours = hours_per_day * days_per_month
    scaled = ResourceSpec(
        cpu_cores=spec.cpu_cores,
        memory_gb=spec.memory_gb,
        gpu_count=spec.gpu_count,
        duration_hours=monthly_hours,
    )
    return estimate_cost(scaled, **rate_kwargs)


def compare_specs(
    specs: list[ResourceSpec],
    labels: list[str] | None = None,
    **rate_kwargs,
) -> list[dict]:
    """Estimate and compare costs for multiple resource configurations.

    Args:
        specs: List of resource specs to compare.
        labels: Optional human-readable labels for each spec.
        **rate_kwargs: Forwarded to :func:`estimate_cost`.

    Returns:
        List of dicts with label and cost breakdown.
    """
    labels = labels or [f"spec-{i}" for i in range(len(specs))]
    results = []
    for label, spec in zip(labels, specs, strict=False):
        breakdown = estimate_cost(spec, **rate_kwargs)
        row = {"label": label, **breakdown.to_dict()}
        results.append(row)
    results.sort(key=lambda r: r["total_usd"])
    return results


__all__ = [
    "CostBreakdown",
    "ResourceSpec",
    "compare_specs",
    "estimate_cost",
    "monthly_estimate",
]
