"""On-site solar generation modelling and self-consumption analysis.

Estimates PV output from array size and irradiance, then works out how much
of that generation a building actually uses on site versus exports to the
grid — the split that drives the economics of a rooftop system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_PANEL_EFFICIENCY: float = 0.20
"""Typical monocrystalline module efficiency."""

DEFAULT_PERFORMANCE_RATIO: float = 0.80
"""System losses: inverter, wiring, soiling, temperature derate."""

DEFAULT_EXPORT_RATE: float = 0.05
"""Payment received per kWh exported to the grid."""


@dataclass
class SolarEconomics:
    """Self-consumption split and value for one generation period."""

    generated_kwh: float
    consumed_kwh: float
    self_consumed_kwh: float
    exported_kwh: float
    imported_kwh: float
    self_consumption_rate: float
    self_sufficiency_rate: float
    bill_saving: float
    export_revenue: float
    total_benefit: float


def generation_kwh(
    array_area_m2: float,
    irradiance_kwh_per_m2: float,
    panel_efficiency: float = DEFAULT_PANEL_EFFICIENCY,
    performance_ratio: float = DEFAULT_PERFORMANCE_RATIO,
) -> float:
    """Estimate PV generation for a period.

    Args:
        array_area_m2: Total module area in square metres.
        irradiance_kwh_per_m2: Plane-of-array irradiance over the period.
        panel_efficiency: Module conversion efficiency in (0, 1].
        performance_ratio: System-level derate factor in (0, 1].

    Returns:
        Generated energy in kWh rounded to 4 decimal places.

    Raises:
        ValueError: If any argument is negative, or either efficiency term
            is outside (0, 1].
    """
    if array_area_m2 < 0 or irradiance_kwh_per_m2 < 0:
        raise ValueError(
            f"area and irradiance must be non-negative, got area={array_area_m2} irradiance={irradiance_kwh_per_m2}"
        )
    if not 0.0 < panel_efficiency <= 1.0:
        raise ValueError(f"panel_efficiency must be in (0, 1], got {panel_efficiency}")
    if not 0.0 < performance_ratio <= 1.0:
        raise ValueError(f"performance_ratio must be in (0, 1], got {performance_ratio}")

    output = array_area_m2 * irradiance_kwh_per_m2 * panel_efficiency * performance_ratio
    return round(output, 4)


def self_consumption(
    generation_hourly_kwh: list[float],
    consumption_hourly_kwh: list[float],
) -> tuple[float, float, float]:
    """Split generation into self-consumed, exported, and imported energy.

    Matching is done hour by hour: solar can only offset load occurring at
    the same time, so a daily total would overstate self-consumption.

    Args:
        generation_hourly_kwh: PV output per hour.
        consumption_hourly_kwh: Site demand per hour.

    Returns:
        Tuple of ``(self_consumed_kwh, exported_kwh, imported_kwh)``, each
        rounded to 4 decimal places.

    Raises:
        ValueError: If the two series differ in length or contain negatives.
    """
    if len(generation_hourly_kwh) != len(consumption_hourly_kwh):
        raise ValueError(
            f"generation and consumption must be the same length "
            f"(got {len(generation_hourly_kwh)} vs {len(consumption_hourly_kwh)})"
        )

    self_consumed = exported = imported = 0.0
    for hour, (gen, load) in enumerate(zip(generation_hourly_kwh, consumption_hourly_kwh, strict=True)):
        if gen < 0 or load < 0:
            raise ValueError(f"values must be non-negative, got generation={gen} consumption={load} at hour {hour}")
        matched = min(gen, load)
        self_consumed += matched
        exported += gen - matched
        imported += load - matched

    return round(self_consumed, 4), round(exported, 4), round(imported, 4)


def analyze_economics(
    generation_hourly_kwh: list[float],
    consumption_hourly_kwh: list[float],
    import_rate: float = 0.15,
    export_rate: float = DEFAULT_EXPORT_RATE,
) -> SolarEconomics:
    """Value a period of solar generation against site demand.

    Args:
        generation_hourly_kwh: PV output per hour.
        consumption_hourly_kwh: Site demand per hour.
        import_rate: Price per kWh paid for grid imports — the rate that
            self-consumed solar avoids.
        export_rate: Payment received per kWh exported.

    Returns:
        A populated :class:`SolarEconomics`.

    Raises:
        ValueError: If the series differ in length, contain negatives, or
            either rate is negative.
    """
    if import_rate < 0 or export_rate < 0:
        raise ValueError(f"rates must be non-negative, got import={import_rate} export={export_rate}")

    self_consumed, exported, imported = self_consumption(generation_hourly_kwh, consumption_hourly_kwh)
    generated = round(sum(generation_hourly_kwh), 4)
    consumed = round(sum(consumption_hourly_kwh), 4)

    self_consumption_rate = round(self_consumed / generated, 4) if generated > 0 else 0.0
    self_sufficiency_rate = round(self_consumed / consumed, 4) if consumed > 0 else 0.0
    bill_saving = round(self_consumed * import_rate, 2)
    export_revenue = round(exported * export_rate, 2)

    logger.info(
        "Solar: generated %.2f kWh, %.1f%% self-consumed, covering %.1f%% of load",
        generated,
        100 * self_consumption_rate,
        100 * self_sufficiency_rate,
    )
    return SolarEconomics(
        generated_kwh=generated,
        consumed_kwh=consumed,
        self_consumed_kwh=self_consumed,
        exported_kwh=exported,
        imported_kwh=imported,
        self_consumption_rate=self_consumption_rate,
        self_sufficiency_rate=self_sufficiency_rate,
        bill_saving=bill_saving,
        export_revenue=export_revenue,
        total_benefit=round(bill_saving + export_revenue, 2),
    )


def payback_years(
    system_cost: float,
    annual_benefit: float,
    annual_degradation: float = 0.005,
) -> float:
    """Return the simple payback period for a PV system.

    Args:
        system_cost: Installed cost of the system.
        annual_benefit: First-year benefit (bill saving plus export revenue).
        annual_degradation: Fractional output loss per year in [0, 1).

    Returns:
        Payback in years rounded to 2 decimal places. Returns ``float('inf')``
        when the benefit never repays the cost.

    Raises:
        ValueError: If *system_cost* is negative, *annual_benefit* is
            negative, or *annual_degradation* is outside [0, 1).
    """
    if system_cost < 0:
        raise ValueError(f"system_cost must be non-negative, got {system_cost}")
    if annual_benefit < 0:
        raise ValueError(f"annual_benefit must be non-negative, got {annual_benefit}")
    if not 0.0 <= annual_degradation < 1.0:
        raise ValueError(f"annual_degradation must be in [0, 1), got {annual_degradation}")
    if annual_benefit == 0:
        return float("inf")

    cumulative = 0.0
    benefit = annual_benefit
    for year in range(1, 51):
        previous = cumulative
        cumulative += benefit
        if cumulative >= system_cost:
            shortfall = system_cost - previous
            fraction = shortfall / benefit if benefit > 0 else 0.0
            return round(year - 1 + fraction, 2)
        benefit *= 1.0 - annual_degradation

    logger.warning("System cost %.2f not repaid within 50 years", system_cost)
    return float("inf")


__all__ = [
    "DEFAULT_EXPORT_RATE",
    "DEFAULT_PANEL_EFFICIENCY",
    "DEFAULT_PERFORMANCE_RATIO",
    "SolarEconomics",
    "analyze_economics",
    "generation_kwh",
    "payback_years",
    "self_consumption",
]
