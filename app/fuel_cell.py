"""Hydrogen fuel cell efficiency and power output modelling."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FARADAY_CONSTANT: float = 96485.0
"""Faraday constant in C/mol."""

HYDROGEN_HHV_KJ_PER_MOL: float = 286.0
"""Higher heating value of hydrogen in kJ/mol."""

HYDROGEN_LHV_KJ_PER_MOL: float = 241.8
"""Lower heating value of hydrogen in kJ/mol."""

STANDARD_CELL_VOLTAGE: float = 1.229
"""Reversible open-circuit voltage of H2/O2 cell at STP in Volts."""

KJ_PER_KWH: float = 3600.0
"""Kilojoules per kilowatt-hour."""


@dataclass
class FuelCellResult:
    """Operating point of a hydrogen fuel cell stack."""

    cell_voltage_v: float
    stack_voltage_v: float
    current_a: float
    power_kw: float
    efficiency_hhv: float
    efficiency_lhv: float
    hydrogen_flow_mol_per_s: float
    heat_generated_kw: float


def nernst_voltage(
    temperature_k: float,
    partial_pressure_h2: float = 1.0,
    partial_pressure_o2: float = 0.21,
    partial_pressure_h2o: float = 1.0,
) -> float:
    """Compute the reversible (Nernst) cell voltage.

    Args:
        temperature_k: Cell temperature in Kelvin.
        partial_pressure_h2: Partial pressure of H2 (atm, default 1.0).
        partial_pressure_o2: Partial pressure of O2 (atm, default 0.21 for air).
        partial_pressure_h2o: Partial pressure of H2O (atm, default 1.0).

    Returns:
        Reversible cell voltage in Volts.

    Raises:
        ValueError: If any pressure is non-positive or temperature <= 0.
    """
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    for p, name in [
        (partial_pressure_h2, "H2"),
        (partial_pressure_o2, "O2"),
        (partial_pressure_h2o, "H2O"),
    ]:
        if p <= 0:
            raise ValueError(f"partial pressure of {name} must be positive")
    R = 8.314
    n = 2
    delta_g0 = -237100.0
    e_rev = -delta_g0 / (n * FARADAY_CONSTANT)
    nernst_correction = (
        (R * temperature_k)
        / (n * FARADAY_CONSTANT)
        * math.log((partial_pressure_h2 * math.sqrt(partial_pressure_o2)) / partial_pressure_h2o)
    )
    return round(e_rev + nernst_correction, 6)


def activation_overpotential(
    current_density_a_per_cm2: float,
    exchange_current_density: float = 1e-4,
    transfer_coefficient: float = 0.5,
    temperature_k: float = 353.15,
) -> float:
    """Butler-Volmer activation overpotential.

    Args:
        current_density_a_per_cm2: Operating current density in A/cm².
        exchange_current_density: Exchange current density in A/cm².
        transfer_coefficient: Charge transfer coefficient (0-1).
        temperature_k: Temperature in Kelvin.

    Returns:
        Activation overpotential in Volts (positive).

    Raises:
        ValueError: If current_density or exchange_current_density <= 0.
    """
    if current_density_a_per_cm2 <= 0:
        raise ValueError("current_density_a_per_cm2 must be positive")
    if exchange_current_density <= 0:
        raise ValueError("exchange_current_density must be positive")
    R = 8.314
    n = 2
    return round(
        (R * temperature_k)
        / (transfer_coefficient * n * FARADAY_CONSTANT)
        * math.log(current_density_a_per_cm2 / exchange_current_density),
        6,
    )


def ohmic_overpotential(
    current_a: float,
    internal_resistance_ohm: float,
) -> float:
    """Compute ohmic voltage loss in the cell.

    Args:
        current_a: Operating current in Amperes.
        internal_resistance_ohm: Cell internal resistance in Ohms.

    Returns:
        Ohmic overpotential in Volts.

    Raises:
        ValueError: If current or resistance is negative.
    """
    if current_a < 0:
        raise ValueError("current_a must be non-negative")
    if internal_resistance_ohm < 0:
        raise ValueError("internal_resistance_ohm must be non-negative")
    return round(current_a * internal_resistance_ohm, 6)


def cell_voltage(
    open_circuit_v: float,
    activation_loss_v: float,
    ohmic_loss_v: float,
    concentration_loss_v: float = 0.0,
) -> float:
    """Net operating cell voltage.

    Args:
        open_circuit_v: Reversible/Nernst voltage in Volts.
        activation_loss_v: Activation overpotential in Volts.
        ohmic_loss_v: Ohmic loss in Volts.
        concentration_loss_v: Concentration/mass-transport loss in Volts.

    Returns:
        Cell operating voltage in Volts (floored at 0).
    """
    v = open_circuit_v - activation_loss_v - ohmic_loss_v - concentration_loss_v
    return round(max(0.0, v), 6)


def stack_power(
    cell_voltage_v: float,
    current_a: float,
    num_cells: int,
) -> float:
    """Compute stack electrical output power.

    Args:
        cell_voltage_v: Individual cell voltage in Volts.
        current_a: Stack current in Amperes.
        num_cells: Number of cells in series.

    Returns:
        Stack power in kW.

    Raises:
        ValueError: If any value is non-positive.
    """
    if cell_voltage_v <= 0:
        raise ValueError("cell_voltage_v must be positive")
    if current_a <= 0:
        raise ValueError("current_a must be positive")
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    power_w = cell_voltage_v * num_cells * current_a
    return round(power_w / 1000.0, 6)


def fuel_cell_efficiency(
    cell_voltage_v: float,
    use_hhv: bool = True,
) -> float:
    """Voltage efficiency of a hydrogen fuel cell.

    Args:
        cell_voltage_v: Operating cell voltage in Volts.
        use_hhv: If True use HHV reference (default), else LHV.

    Returns:
        Efficiency as a fraction in [0, 1].

    Raises:
        ValueError: If cell_voltage_v <= 0.
    """
    if cell_voltage_v <= 0:
        raise ValueError("cell_voltage_v must be positive")
    reference_v = 1.482 if use_hhv else 1.253
    return round(min(1.0, cell_voltage_v / reference_v), 6)


def hydrogen_consumption_rate(
    power_kw: float,
    efficiency: float,
) -> float:
    """Mass flow of hydrogen required to sustain a given power output.

    Args:
        power_kw: Net electrical output in kW.
        efficiency: Stack efficiency (fraction).

    Returns:
        Hydrogen flow in mol/s.

    Raises:
        ValueError: If power_kw or efficiency is non-positive.
    """
    if power_kw <= 0:
        raise ValueError("power_kw must be positive")
    if efficiency <= 0:
        raise ValueError("efficiency must be positive")
    power_kj_per_s = power_kw
    h2_energy_kj_per_mol = HYDROGEN_LHV_KJ_PER_MOL
    mol_per_s = power_kj_per_s / (efficiency * h2_energy_kj_per_mol)
    return round(mol_per_s, 8)


def operating_point(
    current_a: float,
    num_cells: int,
    active_area_cm2: float = 100.0,
    temperature_k: float = 353.15,
    internal_resistance_ohm: float = 0.001,
) -> FuelCellResult:
    """Compute a complete fuel cell stack operating point.

    Args:
        current_a: Stack current in Amperes.
        num_cells: Number of cells in the stack.
        active_area_cm2: Active electrode area per cell in cm².
        temperature_k: Stack temperature in Kelvin.
        internal_resistance_ohm: Cell area-specific resistance in Ω·cm² converted per cell.

    Returns:
        Populated :class:`FuelCellResult`.

    Raises:
        ValueError: If current, cells, or area are non-positive.
    """
    if current_a <= 0:
        raise ValueError("current_a must be positive")
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    if active_area_cm2 <= 0:
        raise ValueError("active_area_cm2 must be positive")

    j = current_a / active_area_cm2
    e_rev = nernst_voltage(temperature_k)
    act_loss = activation_overpotential(j, temperature_k=temperature_k)
    ohm_loss = ohmic_overpotential(current_a, internal_resistance_ohm)
    v_cell = cell_voltage(e_rev, act_loss, ohm_loss)
    p_kw = stack_power(v_cell, current_a, num_cells)
    eff_hhv = fuel_cell_efficiency(v_cell, use_hhv=True)
    eff_lhv = fuel_cell_efficiency(v_cell, use_hhv=False)
    h2_flow = hydrogen_consumption_rate(p_kw, eff_lhv) if p_kw > 0 else 0.0
    heat_kw = round(p_kw * (1.0 - eff_lhv) / eff_lhv, 6) if eff_lhv > 0 else 0.0

    return FuelCellResult(
        cell_voltage_v=v_cell,
        stack_voltage_v=round(v_cell * num_cells, 4),
        current_a=current_a,
        power_kw=p_kw,
        efficiency_hhv=eff_hhv,
        efficiency_lhv=eff_lhv,
        hydrogen_flow_mol_per_s=h2_flow,
        heat_generated_kw=heat_kw,
    )


def degradation_rate_per_hour(
    operating_hours: float,
    voltage_drop_mv: float = 1.0,
) -> float:
    """Return the average cell voltage degradation rate in V/h.

    Args:
        operating_hours: Total hours the cell has operated (must be positive).
        voltage_drop_mv: Total observed voltage drop in mV over the period.

    Returns:
        Degradation rate in V per operating hour.

    Raises:
        ValueError: If operating_hours is non-positive.
    """
    if operating_hours <= 0:
        raise ValueError("operating_hours must be positive")
    return (voltage_drop_mv / 1000.0) / operating_hours


def water_production_rate(hydrogen_flow_mol_per_s: float) -> float:
    """Return the water production rate in mol/s from H2 consumption.

    Each mole of H2 reacted produces one mole of H2O (2H2 + O2 → 2H2O,
    so mol H2O = mol H2 consumed).

    Args:
        hydrogen_flow_mol_per_s: Hydrogen consumption rate in mol/s (non-negative).

    Returns:
        Water production rate in mol/s.

    Raises:
        ValueError: If hydrogen_flow_mol_per_s is negative.
    """
    if hydrogen_flow_mol_per_s < 0:
        raise ValueError("hydrogen_flow_mol_per_s must be non-negative")
    return hydrogen_flow_mol_per_s


def stack_voltage(cell_voltage_v: float, num_cells: int) -> float:
    """Return total stack voltage for a series-connected cell stack.

    Args:
        cell_voltage_v: Single cell voltage in V.
        num_cells: Number of cells in the stack.

    Returns:
        Stack voltage in V.

    Raises:
        ValueError: If cell_voltage_v or num_cells are non-positive.
    """
    if cell_voltage_v <= 0:
        raise ValueError("cell_voltage_v must be positive")
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")
    return cell_voltage_v * num_cells
