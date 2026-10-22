"""Combined Heat and Power (CHP / cogeneration) efficiency modelling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

KJ_PER_KWH: float = 3600.0
"""Kilojoules per kilowatt-hour."""

NATURAL_GAS_KWH_PER_M3: float = 10.55
"""Typical energy content of natural gas in kWh per cubic metre (GCV)."""

GRID_EMISSION_FACTOR_KG_CO2_PER_KWH: float = 0.233
"""Average grid electricity CO2 intensity in kg/kWh."""

GAS_EMISSION_FACTOR_KG_CO2_PER_KWH: float = 0.184
"""Natural gas CO2 emission factor in kg/kWh (combustion only)."""


@dataclass
class CHPResult:
    """Output of a CHP system analysis."""

    fuel_input_kw: float
    electrical_output_kw: float
    heat_output_kw: float
    electrical_efficiency: float
    thermal_efficiency: float
    total_efficiency: float
    power_to_heat_ratio: float
    annual_electrical_mwh: float
    annual_heat_mwh: float
    annual_co2_savings_kg: float
    annual_fuel_cost: float


def chp_electrical_efficiency(
    electrical_output_kw: float,
    fuel_input_kw: float,
) -> float:
    """Electrical efficiency of a CHP unit.

    Args:
        electrical_output_kw: Electrical power output in kW.
        fuel_input_kw: Fuel energy input rate in kW.

    Returns:
        Electrical efficiency as a fraction in (0, 1].

    Raises:
        ValueError: If either value is non-positive or output exceeds input.
    """
    if electrical_output_kw <= 0:
        raise ValueError("electrical_output_kw must be positive")
    if fuel_input_kw <= 0:
        raise ValueError("fuel_input_kw must be positive")
    if electrical_output_kw > fuel_input_kw:
        raise ValueError("electrical output cannot exceed fuel input")
    return round(electrical_output_kw / fuel_input_kw, 6)


def chp_thermal_efficiency(
    heat_output_kw: float,
    fuel_input_kw: float,
) -> float:
    """Thermal efficiency of a CHP unit.

    Args:
        heat_output_kw: Useful heat output in kW.
        fuel_input_kw: Fuel energy input rate in kW.

    Returns:
        Thermal efficiency as a fraction in (0, 1].

    Raises:
        ValueError: If heat output exceeds fuel input or either is non-positive.
    """
    if heat_output_kw <= 0:
        raise ValueError("heat_output_kw must be positive")
    if fuel_input_kw <= 0:
        raise ValueError("fuel_input_kw must be positive")
    if heat_output_kw > fuel_input_kw:
        raise ValueError("heat output cannot exceed fuel input")
    return round(heat_output_kw / fuel_input_kw, 6)


def total_cogeneration_efficiency(
    electrical_output_kw: float,
    heat_output_kw: float,
    fuel_input_kw: float,
) -> float:
    """Total (combined) cogeneration efficiency.

    Args:
        electrical_output_kw: Electrical output in kW.
        heat_output_kw: Thermal output in kW.
        fuel_input_kw: Fuel input in kW.

    Returns:
        Total efficiency as a fraction.

    Raises:
        ValueError: If combined output exceeds fuel input or any value is non-positive.
    """
    combined = electrical_output_kw + heat_output_kw
    if combined > fuel_input_kw * 1.001:
        raise ValueError("Combined outputs exceed fuel input (conservation violation)")
    for val, name in [
        (electrical_output_kw, "electrical"),
        (heat_output_kw, "heat"),
        (fuel_input_kw, "fuel"),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be positive")
    return round(combined / fuel_input_kw, 6)


def co2_savings(
    annual_electrical_mwh: float,
    annual_heat_mwh: float,
    chp_gas_efficiency: float = 0.85,
    grid_emission_factor: float = GRID_EMISSION_FACTOR_KG_CO2_PER_KWH,
    gas_emission_factor: float = GAS_EMISSION_FACTOR_KG_CO2_PER_KWH,
) -> float:
    """CO2 savings of CHP vs. separate heat and power generation.

    Args:
        annual_electrical_mwh: Annual electricity produced by CHP in MWh.
        annual_heat_mwh: Annual heat produced by CHP in MWh.
        chp_gas_efficiency: Thermal efficiency for the heat portion.
        grid_emission_factor: Grid electricity CO2 factor in kg/kWh.
        gas_emission_factor: Gas boiler CO2 factor in kg/kWh.

    Returns:
        Annual CO2 savings in kg.

    Raises:
        ValueError: If energy values are negative.
    """
    if annual_electrical_mwh < 0:
        raise ValueError("annual_electrical_mwh must be non-negative")
    if annual_heat_mwh < 0:
        raise ValueError("annual_heat_mwh must be non-negative")
    grid_co2 = annual_electrical_mwh * 1000 * grid_emission_factor
    boiler_co2 = annual_heat_mwh * 1000 * gas_emission_factor / chp_gas_efficiency
    chp_gas_kwh = (annual_electrical_mwh + annual_heat_mwh) * 1000
    chp_co2 = chp_gas_kwh * gas_emission_factor
    return round(grid_co2 + boiler_co2 - chp_co2, 2)


def analyse_chp(
    fuel_input_kw: float,
    electrical_efficiency: float,
    thermal_efficiency: float,
    capacity_factor: float = 0.80,
    gas_price_per_kwh: float = 0.065,
    electricity_price_per_kwh: float = 0.28,
    heat_price_per_kwh: float = 0.06,
) -> CHPResult:
    """Full CHP system analysis.

    Args:
        fuel_input_kw: Fuel consumption rate in kW.
        electrical_efficiency: Electrical efficiency fraction.
        thermal_efficiency: Thermal efficiency fraction.
        capacity_factor: Annual plant capacity factor.
        gas_price_per_kwh: Gas price in $/kWh.
        electricity_price_per_kwh: Grid electricity price in $/kWh.
        heat_price_per_kwh: District heat or equivalent value in $/kWh.

    Returns:
        Populated :class:`CHPResult`.

    Raises:
        ValueError: If efficiencies sum > 1 or any value is out of range.
    """
    if fuel_input_kw <= 0:
        raise ValueError("fuel_input_kw must be positive")
    if not (0 < electrical_efficiency < 1):
        raise ValueError("electrical_efficiency must be in (0, 1)")
    if not (0 < thermal_efficiency < 1):
        raise ValueError("thermal_efficiency must be in (0, 1)")
    if electrical_efficiency + thermal_efficiency > 1.0:
        raise ValueError("Sum of efficiencies cannot exceed 1.0")

    elec_kw = fuel_input_kw * electrical_efficiency
    heat_kw = fuel_input_kw * thermal_efficiency
    total_eff = electrical_efficiency + thermal_efficiency
    phr = elec_kw / heat_kw if heat_kw > 0 else 0.0
    hours = 8760 * capacity_factor
    annual_elec = elec_kw * hours / 1000.0
    annual_heat = heat_kw * hours / 1000.0
    annual_fuel_cost = fuel_input_kw * hours * gas_price_per_kwh / 1000.0
    co2_sav = co2_savings(annual_elec, annual_heat)

    return CHPResult(
        fuel_input_kw=round(fuel_input_kw, 4),
        electrical_output_kw=round(elec_kw, 4),
        heat_output_kw=round(heat_kw, 4),
        electrical_efficiency=round(electrical_efficiency, 6),
        thermal_efficiency=round(thermal_efficiency, 6),
        total_efficiency=round(total_eff, 6),
        power_to_heat_ratio=round(phr, 4),
        annual_electrical_mwh=round(annual_elec, 4),
        annual_heat_mwh=round(annual_heat, 4),
        annual_co2_savings_kg=co2_sav,
        annual_fuel_cost=round(annual_fuel_cost, 2),
    )


def heat_to_power_ratio(thermal_output_kw: float, electrical_output_kw: float) -> float:
    """Return the heat-to-power ratio (HPR) of a CHP system.

    HPR > 1 means more heat than power is produced; HPR < 1 indicates
    power-dominated systems.

    Args:
        thermal_output_kw: Useful thermal output in kW (non-negative).
        electrical_output_kw: Electrical output in kW (must be positive).

    Returns:
        Heat-to-power ratio (dimensionless).

    Raises:
        ValueError: If electrical_output_kw ≤ 0 or thermal_output_kw < 0.
    """
    if electrical_output_kw <= 0:
        raise ValueError("electrical_output_kw must be positive")
    if thermal_output_kw < 0:
        raise ValueError("thermal_output_kw must be non-negative")
    return thermal_output_kw / electrical_output_kw


def primary_energy_saving(
    chp_fuel_input_kw: float,
    reference_electrical_efficiency: float = 0.40,
    reference_thermal_efficiency: float = 0.90,
    electrical_output_kw: float = 0.0,
    thermal_output_kw: float = 0.0,
) -> float:
    """Return the Primary Energy Saving (PES) factor for a CHP system.

    PES = 1 - 1 / (E/(η_e_ref) + H/(η_h_ref)) / Q_chp, where E and H
    are electrical and thermal outputs and η refs are reference efficiencies.
    Simplified here to compare against separate generation.

    Args:
        chp_fuel_input_kw: Total fuel input to the CHP system in kW.
        reference_electrical_efficiency: Reference power-only efficiency (0–1).
        reference_thermal_efficiency: Reference heat-only efficiency (0–1).
        electrical_output_kw: CHP electrical output in kW.
        thermal_output_kw: CHP useful heat output in kW.

    Returns:
        Primary energy saving factor (positive means primary energy saved).

    Raises:
        ValueError: If chp_fuel_input_kw ≤ 0 or reference efficiencies invalid.
    """
    if chp_fuel_input_kw <= 0:
        raise ValueError("chp_fuel_input_kw must be positive")
    if not 0 < reference_electrical_efficiency <= 1:
        raise ValueError("reference_electrical_efficiency must be in (0, 1]")
    if not 0 < reference_thermal_efficiency <= 1:
        raise ValueError("reference_thermal_efficiency must be in (0, 1]")

    ref_fuel = (
        electrical_output_kw / reference_electrical_efficiency
        + thermal_output_kw / reference_thermal_efficiency
    )
    if ref_fuel == 0:
        return 0.0
    return 1.0 - chp_fuel_input_kw / ref_fuel


def annual_fuel_cost(
    fuel_input_kw: float,
    operating_hours: float,
    fuel_price_per_kwh: float,
) -> float:
    """Return the annual fuel cost for a CHP system.

    Args:
        fuel_input_kw: Fuel input power in kW (must be positive).
        operating_hours: Annual operating hours (must be positive).
        fuel_price_per_kwh: Fuel price in currency per kWh thermal.

    Returns:
        Annual fuel cost in currency units.

    Raises:
        ValueError: If any argument is non-positive.
    """
    if fuel_input_kw <= 0:
        raise ValueError("fuel_input_kw must be positive")
    if operating_hours <= 0:
        raise ValueError("operating_hours must be positive")
    if fuel_price_per_kwh <= 0:
        raise ValueError("fuel_price_per_kwh must be positive")
    return fuel_input_kw * operating_hours * fuel_price_per_kwh
